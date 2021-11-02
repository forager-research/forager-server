import asyncio
import os
import os.path
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from forager.utils import create_unlimited_aiohttp_session, download_multiple

PATHS_FILE = os.path.join(os.path.dirname(__file__), "demo_dataset_paths.txt")
EMBEDDINGS_PATH = "https://storage.googleapis.com/foragerml/waymo_demo_embeddings.npy"
LABELS_PATH = "N/A"
ARCHIVE_PATH = "https://storage.googleapis.com/foragerml/waymo_demo_dataset.tar.gz"


def get_paths() -> List[str]:
    with open(PATHS_FILE, "r") as f:
        return [x.strip() for x in f.readlines()]


def get_embeddings() -> List[str]:
    with open(PATHS_FILE, "r") as f:
        return [x.strip() for x in f.readlines()]


def get_archive_path() -> str:
    """Archive format:

    train_images/*.jpg
    val_images/*.jpg
    model_outputs/{model_name}/embeddings.bin
    model_outputs/{model_name}/image_list.txt
    image_list.txt
    """
    return ARCHIVE_PATH


async def generate_paths():
    GCS_FOLDER = "gs://foragerml/waymo/train/"
    URL_PREFIX = "https://storage.googleapis.com/foragerml/waymo/train/"
    proc = await asyncio.create_subprocess_exec(
        "gsutil",
        "ls",
        GCS_FOLDER,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    paths = []
    for line in stdout.decode("UTF-8").split("\n"):
        if ".jpeg" not in line:
            continue
        suffix = os.path.basename(line)
        paths.append(os.path.join(URL_PREFIX, suffix))
    total_images = len(paths)
    print("Num images: ", total_images)
    # Select random 1000 image subset
    target_images = 1000
    stride = int(total_images / target_images)
    print(paths[::stride])
    with open(PATHS_FILE, "w") as f:
        f.write("\n".join(paths[::stride]))


async def add_initial_dataset():
    http_session = create_unlimited_aiohttp_session()
    # DEMO_DATASET_DIR = Path("~/.forager/staging_dataset").expanduser()
    # # Download images
    # urls = get_paths()
    # output_directory = str(DEMO_DATASET_DIR)
    # print("Downloading demo dataset images...")
    # os.makedirs(output_directory, exist_ok=True)
    # await download_multiple(http_session, urls, output_directory)
    # # Download pre-computed embeddings
    # # Add to db
    # request = dict(
    #     dataset="demo_dataset",
    #     train_images_directory=output_directory,
    #     val_images_directory="",
    # )
    # print("Adding demo dataset to Forager...")
    # async with http_session.post(
    #     f"http://localhost:8000/api/create_dataset",
    #     json=request,
    #     timeout=1200,
    # ) as response:
    #     print(response)

    job_ids = []
    request = {
        "dataset": "demo_dataset",
        "model": "resnet",
        "model_output_name": "resnet",
    }
    async with http_session.post(
        f"http://localhost:8000/api/start_model_inference",
        json=request,
        timeout=1200,
    ) as response:
        print(response)
        data = await response.json()
        job_ids.append(data["job_id"])

    request = {
        "dataset": "demo_dataset",
        "model": "clip",
        "model_output_name": "clip",
    }
    async with http_session.post(
        f"http://localhost:8000/api/start_model_inference",
        json=request,
        timeout=1200,
    ) as response:
        data = await response.json()
        job_ids.append(data["job_id"])

    not_done = True
    while not_done:
        not_done = False
        request = {"job_ids": job_ids}
        async with http_session.get(
            f"http://localhost:8000/api/model_inference_status",
            json=request,
        ) as response:
            data = await response.json()
            for job_id, info in data.items():
                if not info["finished"]:
                    not_done = True
                    continue

    print("Success! Dataset added as 'demo_dataset'.")


async def export_dataset(
    backend_url: str, forager_data_dir: str, dataset_name: str, archive_path: str
) -> bool:
    http_session = create_unlimited_aiohttp_session()
    # Get model outputs
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_outputs_archive_dir = os.path.join(tmpdirname, "model_outputs")
        os.makedirs(model_outputs_archive_dir)
        train_images_archive_dir = os.path.join(tmpdirname, "train_images")
        val_images_archive_dir = os.path.join(tmpdirname, "val_images")
        # Get images
        params = {"extended_info": "yes"}
        async with http_session.get(
            f"{backend_url}/api/get_dataset_info/{dataset_name}",
            params=params,
        ) as response:
            if response.status != 200:
                print("Error!", response)
                return False
            data = await response.json()
            print(data)
            train_images_directory = data["train_directory"]
            val_images_directory = data["val_directory"]
        # Copy images into temp directory
        if train_images_directory:
            shutil.copytree(train_images_directory, train_images_archive_dir)
        if val_images_directory:
            shutil.copytree(val_images_directory, val_images_archive_dir)

        # Get model outputs
        async with http_session.get(
            f"{backend_url}/api/get_model_outputs/{dataset_name}",
            params={"extended_info": "yes"},
        ) as response:
            if response.status != 200:
                print("Error!", response)
                return False
            data = await response.json()
            for model_output in data["model_outputs"]:
                print(f'Exporting {model_output["name"]} model output...')
                print(model_output)
                this_output_dir = os.path.join(
                    model_outputs_archive_dir, model_output["name"]
                )
                os.makedirs(this_output_dir)
                image_list_path = model_output["image_list_path"]
                # Remap paths to be local to archive
                remapped_paths = []
                with open(image_list_path, "r") as f:
                    for line in f.readlines():
                        split, ident, path = line.split()
                        if split == "train":
                            new_path = os.path.join(
                                "train_images", os.path.basename(path)
                            )
                        else:
                            new_path = os.path.join(
                                "val_images", os.path.basename(path)
                            )
                        remapped_paths.append(new_path)
                    num_images = len(remapped_paths)
                dest_path = os.path.join(this_output_dir, "image_list.txt")
                with open(dest_path, "w") as f:
                    for path in remapped_paths:
                        f.write(f"n/a n/a {path}\n")
                # Write embeddings
                if model_output["has_embeddings"]:
                    embeddings_path = model_output["embeddings_path"]
                    dest_path = os.path.join(this_output_dir, "embeddings.bin")
                    shutil.copy(
                        embeddings_path,
                        dest_path,
                    )
                # Write scores
                if model_output["has_scores"]:
                    scores_path = model_output["scores_path"]
                    dest_path = os.path.join(this_output_dir, "scores.bin")
                    shutil.copy(
                        scores_path,
                        dest_path,
                    )
        # Build archive
        proc = await asyncio.create_subprocess_exec(
            "tar", "-czf", archive_path, ".", cwd=tmpdirname
        )
        await proc.wait()
        print(f"Archive created at: {archive_path}")

        return True


if __name__ == "__main__":
    # asyncio.run(generate_paths())
    # asyncio.run(add_initial_dataset())

    FORAGER_DIR = Path("~/.forager").expanduser()
    archive_path = os.path.join(os.getcwd(), "demo_dataset.tar.gz")
    asyncio.run(
        export_dataset(
            "http://localhost:8000", str(FORAGER_DIR), "demo_dataset", archive_path
        )
    )
