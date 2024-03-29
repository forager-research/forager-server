from __future__ import annotations

import asyncio
import concurrent
import logging
import multiprocessing as mp
import time
import traceback
from typing import Any, Dict, Optional

import forager_embedding_server.models as models
import forager_embedding_server.utils as utils
import forager_knn.utils as knn_utils
import numpy as np
from forager_embedding_server.jobs_data import load_image_list
from PIL import Image

logger = logging.getLogger("index_server")

BUILTIN_MODELS = {
    "clip": models.CLIP,
    "resnet": models.ResNet,
}

_EXECUTOR_MODEL: Optional[models.EmbeddingModel] = None


@utils.trace_unhandled_exceptions
def _executor_init_fn(embedding_type):
    global _EXECUTOR_MODEL
    _EXECUTOR_MODEL = BUILTIN_MODELS[embedding_type]()


@utils.trace_unhandled_exceptions
def _executor_evaluate(images):
    global _EXECUTOR_MODEL
    return _EXECUTOR_MODEL.embed_images(images)


class EmbeddingInferenceJob:
    def __init__(
        self,
        job_id: str,
        model_output_name: str,
        image_list_path: str,
        embedding_type: str,
        output_path: str,
        session: aiohttp.ClientSession,
        callback_url: Optional[str] = None,
        callback_data: Optional[Dict] = None,
    ):
        self.job_id = job_id
        self.model_output_name = model_output_name
        self.image_list_path = image_list_path
        self.embedding_type = embedding_type
        self.embeddings_path = output_path
        self.session = session
        self.callback_url = callback_url
        self.callback_data = callback_data

        self.image_list = load_image_list(image_list_path)

        self.started = False
        self.finished = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure_reason: Optional[str] = None
        self.profiling: Dict[str, float] = {}

        self.batch_size = 8

        self._failed_or_finished = asyncio.Condition()

        # Will be initialized later
        self.job_args: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

        self._time_left: Optional[float] = None
        self._n_processed: Optional[int] = None

    @property
    def status(self):
        if self._task:
            start_time = self._start_time or time.time()
            end_time = self._end_time or time.time()
            prog = {
                "elapsed_time": end_time - start_time,
                "n_processed": self._n_processed or 0,
                "n_total": len(self.image_list),
            }
            total_left = prog["n_total"] - prog["n_processed"]
            if prog["n_processed"] > 0:
                time_left = total_left * (prog["elapsed_time"] / prog["n_processed"])
            else:
                time_left = -1
            return {
                "started": self.started,
                "finished": self.finished.is_set(),
                "failed": self.failed.is_set(),
                "failure_reason": self.failure_reason or "",
                "elapsed_time": end_time - start_time,
                "time_left": time_left,
                "progress": prog,
            }
        else:
            return {
                "started": self.started,
                "finished": self.finished.is_set(),
                "failed": self.failed.is_set(),
                "failure_reason": self.failure_reason or "",
                "elapsed_time": 0,
                "time_left": -1,
            }

    def start(self, executor=None):
        self.started = True
        self._task = self._run_in_background(executor)

    @knn_utils.unasync_as_task
    async def _run_in_background(self, executor):
        self._start_time = time.time()

        async def finish(failure_reason: Optional[str] = None):
            logger.info(
                f"EmbeddingJob[{self.job_id}]: finished with {len(self.image_list)} images"
            )
            self._end_time = time.time()
            if failure_reason:
                logger.info(f"EmbeddingJob[{self.job_id}]: failed: {failure_reason}")
                self.failure_reason = failure_reason
                self.failed.set()

            self.finished.set()

            if self.callback_url:
                status = {
                    "finished": self.status["finished"],
                    "failed": self.status["failed"],
                    "failure_reason": self.status["failure_reason"],
                    "status": self.status,
                    "model_output_name": self.model_output_name,
                    "image_list_path": self.image_list_path,
                    "embeddings_path": self.embeddings_path,
                    "callback_data": self.callback_data,
                }
                async with self.session.post(
                    self.callback_url, json=status
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"EmbeddingJob[{self.job_id}]: callback to {self.callback_url} failed"
                        )

        try:
            logger.info(
                f"EmbeddingJob[{self.job_id}]: started with {len(self.image_list)} images"
            )

            if len(self.image_list) == 0:
                with open(self.embeddings_path, "w") as f:
                    pass
                await finish()

            # Create model in executor
            mp_context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                initializer=_executor_init_fn,
                initargs=(self.embedding_type,),
                mp_context=mp_context,
            ) as executor:
                model_cls = BUILTIN_MODELS[self.embedding_type]
                model_output_dim = model_cls.output_dim()
                embeddings = np.memmap(
                    self.embeddings_path,
                    dtype="float32",
                    mode="w+",
                    shape=(len(self.image_list), model_output_dim),
                )
                BATCH_SIZE = 32
                self._n_processed = 0
                for idx_start in range(0, len(self.image_list), BATCH_SIZE):
                    image_paths = self.image_list.get_paths(
                        range(
                            idx_start, min(idx_start + BATCH_SIZE, len(self.image_list))
                        )
                    )
                    # Read images
                    images = [np.asarray(Image.open(path)) for path in image_paths]
                    # Compute embeddings in executor
                    embeddings[
                        idx_start : idx_start + len(images)
                    ] = await asyncio.get_running_loop().run_in_executor(
                        executor, _executor_evaluate, images
                    )
                    self._n_processed += len(image_paths)
                    await asyncio.sleep(0)

            embeddings.flush()
            await finish()

        except Exception as e:
            print(traceback.print_exc())
            await finish(failure_reason=str(e))

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task
