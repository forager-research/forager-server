import asyncio
from enum import Enum
import os
from pathlib import Path

import aiohttp
import numpy as np
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from typing import List, Optional, Tuple, Union

from knn import utils
from knn.mappers import Mapper

import config
import inference


class IndexEmbeddingMapper(Mapper):
    class ReturnType(Enum):
        SAVE = 0
        SERIALIZE = 1

    def initialize_container(self):
        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(
            build_resnet_backbone(config.RESNET_CONFIG, shape)
        )

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(config.WEIGHTS_PATH)
        self.model.eval()

        # Store relevant attributes of config
        self.pixel_mean = torch.tensor(config.RESNET_CONFIG.MODEL.PIXEL_MEAN).view(
            -1, 1, 1
        )
        self.pixel_std = torch.tensor(config.RESNET_CONFIG.MODEL.PIXEL_STD).view(
            -1, 1, 1
        )
        self.input_format = config.RESNET_CONFIG.INPUT.FORMAT

        # Create connection pool
        self.session = aiohttp.ClientSession()

    async def initialize_job(self, job_args):
        return_type = job_args.get("return_type", "serialize")
        if return_type == "save":
            job_args["return_type"] = IndexEmbeddingMapper.ReturnType.SAVE
        elif return_type == "serialize":
            job_args["return_type"] = IndexEmbeddingMapper.ReturnType.SERIALIZE
        else:
            raise ValueError(f"Unknown return type: {return_type}")

        job_args["n_chunks_saved"] = 0
        return job_args

    @utils.log_exception_from_coro_but_return_none
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> np.ndarray:
        image_path = input["image"]
        image_patch = input.get("patch", (0, 0, 1, 1))
        augmentations = input.get("augmentations", {})

        # Download image
        if "http" not in image_path:
            image_bucket = job_args["input_bucket"]
            image_path = os.path.join(config.GCS_URL_PREFIX, image_bucket, image_path)
        image_bytes = await self.download_image(image_path)

        # Run inference
        with self.profiler(request_id, "inference_time"):
            model_output_dict = inference.run(
                image_bytes,
                image_patch,
                augmentations,
                self.input_format,
                self.pixel_mean,
                self.pixel_std,
                self.model,
            )

        with self.profiler(request_id, "flatten_time"):
            spatial_embeddings = next(iter(model_output_dict.values())).numpy()
            n, c, h, w = spatial_embeddings.shape
            assert n == 1
            return np.ascontiguousarray(spatial_embeddings.reshape((c, h * w)).T)

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

    async def postprocess_chunk(
        self,
        inputs,
        outputs: List[Optional[np.ndarray]],
        job_id,
        job_args,
        request_id,
    ) -> Union[Tuple[str, List[Optional[int]]], Tuple[None, List[Optional[str]]]]:
        if job_args["return_type"] == IndexEmbeddingMapper.ReturnType.SAVE:
            with self.profiler(request_id, "reduce_time"):
                embeddings_dicts = {
                    reduction: {
                        int(input["id"]): reduce_fn(output)
                        for input, output in zip(inputs, outputs)
                        if output is not None
                    }
                    for reduction, reduce_fn in config.REDUCTIONS.items()
                }

            with self.profiler(request_id, "save_time"):
                output_path_tmpl = config.EMBEDDINGS_FILE_TMPL.format(
                    job_id, self.worker_id, job_args["n_chunks_saved"]
                )
                job_args["n_chunks_saved"] += 1
                Path(output_path_tmpl).parent.mkdir(parents=True, exist_ok=True)

                for name, embeddings_dict in embeddings_dicts.items():
                    np.save(output_path_tmpl.format(name), embeddings_dict)

            return output_path_tmpl, [
                len(output) if output is not None else None for output in outputs
            ]
        else:
            with self.profiler(request_id, "reduce_time"):
                reduce_fn = config.REDUCTIONS[job_args.get("reduction")]
                reduced_outputs = [
                    reduce_fn(output) if output is not None else None
                    for output in outputs
                ]

            with self.profiler(request_id, "serialize_time"):
                serialized_outputs = [
                    utils.numpy_to_base64(output) if output is not None else None
                    for output in reduced_outputs
                ]

            return None, serialized_outputs


app = IndexEmbeddingMapper().server
