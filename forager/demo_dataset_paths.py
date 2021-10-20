import asyncio
import os
import os.path
import shutil
from pathlib import Path
from typing import List

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


async def export_dataset(
    backend_url: str, forager_data_dir: str, dataset_name: str, archive_path: str
) -> bool:
    # Get model outputs
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_outputs_archive_dir = os.path.join(tmpdirname, "model_outputs")
        os.makedirs(model_outputs_dir)
        train_images_archive_dir = os.path.join(tmpdirname, "train_images")
        val_images_archive_dir = os.path.join(tmpdirname, "val_images")
        async with http_session.post(
            f"{backend_url}/api/get_model_outputs/{dataset_name}",
            timeout=1200,
        ) as response:
            if response.status != 200:
                print("Error!", response)
                return False
            for model_output in response.json["model_outputs"]:
                print(f'Exporting {model_output["name"]} model output...')
                this_output_dir = os.path.join(
                    model_outputs_archive_dir, model_output["name"]
                )
                os.makedirs(this_output_dir)
                image_lists_path = os.path.join(
                    forager_data_dir, "image_lists", model_output["id"]
                )
                dest_path = os.path.join(this_output_dir, "image_lists.txt")
                shutil.copy(
                    image_lists_path,
                    dest_path,
                )
                if model_output.has_embeddings:
                    embeddings_path = os.path.join(
                        forager_data_dir, "embeddings", model_output["id"]
                    )
                    dest_path = os.path.join(this_output_dir, "embeddings.bin")
                    shutil.copy(
                        embeddings_path,
                        dest_path,
                    )
                if model_output.has_scores:
                    embeddings_path = os.path.join(
                        forager_data_dir, "scores", model_output["id"]
                    )
                    dest_path = os.path.join(this_output_dir, "scores.bin")
                    shutil.copy(
                        embeddings_path,
                        dest_path,
                    )
        # Get images
        params = {"request_extended": "yes"}
        async with http_session.get(
            f"{backend_url}/api/get_dataset_info/{dataset_name}",
            params=params,
        ) as response:
            if response.status != 200:
                print("Error!", response)
                return False
            train_images_directory = response.json["train_directory"]
            val_images_directory = response.json["val_directory"]
        # Copy images into temp directory
        shutil.copytree(train_images_directory, train_images_archive_dir)
        shutil.copytree(val_images_directory, val_images_archive_dir)

        # Build archive
        proc = await asyncio.create_subprocess_exec(
            "tar", "-czf", archive_path, f"{tmpdirname}/*"
        )
        await proc.wait()
        print(f"Archive created at: {archive_path}")

        return True


if __name__ == "__main__":
    # asyncio.run(generate_paths())

    FORAGER_DIR = Path("~/.forager").expanduser()
    archive_path = "demo_dataset.tar.gz"
    asyncio.run(
        export_dataset(
            "http://localhost:8000", str(FORAGER_DIR), "demo_dataset", archive_path
        )
    )
