import asyncio
import io
import os

import aiohttp
from gcloud.aio.storage import Storage
from PIL import Image

from knn import utils
from knn.mappers import Mapper

import config


class ImageResizingMapper(Mapper):
    def initialize_container(self):
        self.session = aiohttp.ClientSession()
        self.storage_client = Storage(session=self.session)

    @utils.log_exception_from_coro_but_return_none
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> str:
        image_path = input["image"]
        identifier = input["identifier"]

        output_bucket = job_args["output_bucket"]
        output_dir = job_args["output_dir"]
        resize_max_width = job_args.get("resize_max_width")
        resize_max_height = job_args.get("resize_max_height")

        assert resize_max_width or resize_max_height
        output_path = os.path.join(output_dir, f"{identifier}.jpg")

        # Download image
        if "http" not in image_path:
            image_bucket = job_args["input_bucket"]
            image_path = os.path.join(config.GCS_URL_PREFIX, image_bucket, image_path)
        image_bytes = await self.download_image(image_path)

        with io.BytesIO(image_bytes) as image_buffer:
            # Resize
            image = Image.open(image_buffer)
            image = image.convert("RGB")
            image.thumbnail(
                (resize_max_width or image.width, resize_max_height or image.height)
            )

            # Upload to Cloud Storage
            with io.BytesIO() as output_buffer:
                image.save(output_buffer, "jpeg")
                output_buffer.seek(0)
                await self.storage_client.upload(
                    output_bucket, output_path, output_buffer
                )

        return os.path.join(config.GCS_URL_PREFIX, output_bucket, output_path)

    async def download_image(
        self, image_path: str, num_retries: int = config.DOWNLOAD_NUM_RETRIES
    ) -> bytes:
        for i in range(num_retries + 1):
            try:
                async with self.session.get(image_path) as response:
                    assert response.status == 200
                    return await response.read()
            except Exception:
                if i < num_retries:
                    await asyncio.sleep(2 ** i)
                else:
                    raise
        assert False  # unreachable


app = ImageResizingMapper().server
