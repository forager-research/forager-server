import asyncio
import os.path
from typing import List

PATHS_FILE = os.path.join(os.path.dirname(__file__), "demo_dataset_paths.txt")
EMBEDDINGS_PATH = "https://storage.googleapis.com/foragerml/waymo_demo_embeddings.npy"


def get_paths() -> List[str]:
    with open(PATHS_FILE, "r") as f:
        return [x.strip() for x in f.readlines()]


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


if __name__ == "__main__":
    asyncio.run(generate_paths())
