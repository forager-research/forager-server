from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import forager_knn.utils as utils
import numpy as np
from PIL import Image

import forager_embedding_server.config as config
import forager_embedding_server.models as models
from forager_embedding_server.jobs_data import ImageList, load_image_list

logger = logging.getLogger("index_server")

BUILTIN_MODELS = {
    "clip": models.CLIP(),
    "resnet": models.ResNet(),
}


class EmbeddingInferenceJob:
    def __init__(
        self,
        job_id: str,
        image_list_path: str,
        embedding_type: str,
        output_path: str,
    ):
        self.job_id = job_id
        self.image_list_path = image_list_path
        self.embedding_type = embedding_type
        self.embeddings_path = output_path

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

    def start(self):
        self.started = True
        self._task = self.run_in_background()

    @utils.unasync_as_task
    async def run_in_background(self):
        self._start_time = time.time()

        def finish(failure_reason: Optional[str] = None):
            logger.info(
                f"EmbeddingJob[{self.job_id}]: finished with {len(self.image_list)} images"
            )
            self._end_time = time.time()
            if failure_reason:
                logger.info(f"EmbeddingJob[{self.job_id}]: failed: {failure_reason}")
                self.failure_reason = failure_reason
                self.failed.set()

            self.finished.set()

        try:
            logger.info(f"EmbeddingJob: started with {len(self.image_list)} images")

            if len(self.image_list) == 0:
                with open(self.embeddings_path, "w") as f:
                    pass
                finish()

            model = BUILTIN_MODELS[self.embedding_type]
            model_output_dim = model.output_dim()
            embeddings = np.memmap(
                self.embeddings_path,
                dtype="float32",
                mode="w+",
                shape=(len(self.image_list), model_output_dim),
            )
            BATCH_SIZE = 8
            for idx_start in range(0, len(self.image_list), BATCH_SIZE):
                image_paths = self.image_list.get_paths(
                    range(idx_start, min(idx_start + BATCH_SIZE, len(self.image_list)))
                )
                # Read images
                images = [np.asarray(Image.open(path)) for path in image_paths]
                # Compute embeddings
                embeddings[idx_start : idx_start + len(images)] = model.embed_images(
                    images
                )

            embeddings.flush()
            finish()

        except Exception as e:
            print(traceback.print_exc())
            finish(failure_reason=str(e))

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task
