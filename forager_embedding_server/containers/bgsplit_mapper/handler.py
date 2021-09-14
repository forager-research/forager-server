import numpy as np
import aiohttp
import asyncio
import os.path
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms, utils, io
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum

from knn import utils
from knn.mappers import Mapper
from knn.utils import JSONType

import config
from model import Model

class BGSplittingMapper(Mapper):
    class ReturnType(Enum):
        SAVE = 0
        SERIALIZE = 1

    def initialize_container(self):
        # Create connection pool
        self.session = aiohttp.ClientSession()
        self.use_cuda = False

    async def initialize_job(self, job_args):
        return_type = job_args.get("return_type", "serialize")
        if return_type == "save":
            job_args["return_type"] = BGSplittingMapper.ReturnType.SAVE
        elif return_type == "serialize":
            job_args["return_type"] = BGSplittingMapper.ReturnType.SERIALIZE
        else:
            raise ValueError(f"Unknown return type: {return_type}")
        # Get checkpoint data
        if job_args["checkpoint_path"] == 'TEST':
            model = Model(num_main_classes=2, num_aux_classes=1)
        else:
            map_location = torch.device('cuda') if self.use_cuda else torch.device('cpu')
            checkpoint_state = torch.load(job_args["checkpoint_path"],
                                          map_location=map_location)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint_state['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            if 'model_kwargs' in checkpoint_state:
                kwargs = checkpoint_state['model_kwargs']
                num_aux_classes = kwargs['num_aux_classes']
            else:
                num_aux_classes = 1

            # Create model
            model = Model(num_main_classes=2, num_aux_classes=num_aux_classes)
            # Load model weights
            model.load_state_dict(new_state_dict)
        model.eval()
        if self.use_cuda:
            model = model.cuda()
        job_args["model"] = model
        job_args["transform"] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        job_args["n_chunks_saved"] = 0
        return job_args

    @utils.log_exception_from_coro_but_return_none
    async def process_chunk(
        self, chunk: List[JSONType], job_id: str, job_args: Any, request_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_paths = [c["path"] for c in chunk]

        # Download images
        if "http" not in image_paths[0]:
            image_bucket = job_args["input_bucket"]
            image_paths = [
                os.path.join(config.GCS_URL_PREFIX, image_bucket, image_path)
                for image_path in image_paths]

        transform = job_args["transform"]
        async def download_transform(image_path):
            return await self.transform_image(
                await self.download_image(image_path),
                transform=transform)
        with self.profiler(request_id, "download_time"):
            input_images = await asyncio.gather(
                *[
                    download_transform(image_path)
                    for image_path in image_paths
                ])

        # Run inference
        model = job_args["model"]
        with self.profiler(request_id, "inference_time"):
            image_batch = torch.stack(input_images)
            if self.use_cuda:
                image_batch = image_batch.cuda()
            embeddings = model.forward_backbone(image_batch)
            scores = F.softmax(model.main_head(embeddings), dim=1)[:, 1]

        return (embeddings.detach().cpu().numpy(),
                scores.detach().cpu().numpy())

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

    async def transform_image(
            self, image_bytes: bytes, transform,
    ) -> torch.Tensor:
        data = torch.tensor(
            list(image_bytes),
            dtype=torch.uint8)
        image = io.decode_image(data, mode=io.image.ImageReadMode.RGB)
        return transform(image)

    async def postprocess_chunk(
        self,
        inputs,
        outputs: Tuple[np.ndarray, np.ndarray],
        job_id,
        job_args,
        request_id,
    ) -> Union[Tuple[str, List[Optional[int]]],
               Tuple[None, List[Optional[str]]]]:
        if job_args["return_type"] == BGSplittingMapper.ReturnType.SAVE:
            with self.profiler(request_id, "save_time"):
                data_path_tmpl = config.DATA_FILE_TMPL.format(
                    job_id, self.worker_id, job_args["n_chunks_saved"]
                )
                job_args["n_chunks_saved"] += 1
                Path(data_path_tmpl).parent.mkdir(parents=True, exist_ok=True)

                data = {'ids': np.array([inp['id'] for inp in inputs], dtype=np.int),
                        'embeddings': outputs[0],
                        'scores': outputs[1]}
                np.save(data_path_tmpl.format(None), data)

            return data_path_tmpl.format(None), [
                len(output) if output is not None else None for output in outputs[0]
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

    async def process_element(
        self,
        input: JSONType,
        job_id: str,
        job_args: Any,
        request_id: str,
        element_index: int,
    ) -> Any:
        pass


app = BGSplittingMapper().server
