import concurrent.futures
from dataclasses import dataclass, field
import functools
import signal
import threading
import time
import traceback

import backoff
import numpy as np
import requests
from flask import Flask, request, abort

from typing import Any, Dict, List, Iterable, Optional, Tuple

from interactive_index import InteractiveIndex

import config


# Step 1: Load saved embeddings into memory
def load(paths: Iterable[str], sample_rate: float) -> Tuple[np.ndarray, int]:
    all_embeddings = []
    num_paths_read = 0
    load_func = functools.partial(load_one, sample_rate=sample_rate)

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for loaded_embeddings in pool.map(load_func, paths):
            if loaded_embeddings:
                all_embeddings.extend(loaded_embeddings)
                num_paths_read += 1

    return np.concatenate(all_embeddings), num_paths_read


def load_one(path: str, sample_rate: float) -> Optional[List[np.ndarray]]:
    try:
        # Each file is a np.save'd Dict[int, np.ndarray] where each value is N x D
        embedding_dict = np.load(
            path, allow_pickle=True
        ).item()  # type: Dict[int, np.ndarray]
    except Exception as e:
        print(f"Error in load (path = {path}), but ignoring. {type(e)}: {e}")
        traceback.print_exc()
        return None

    loaded_embeddings = []
    for embeddings in embedding_dict.values():
        if sample_rate:
            n = embeddings.shape[0]
            n_sample = np.random.binomial(n, sample_rate)
            if n_sample == 0:
                continue
            elif n_sample < n:
                sample_inds = np.random.choice(n, n_sample)
                embeddings = embeddings[sample_inds]
        loaded_embeddings.append(embeddings)

    return loaded_embeddings


# Step 2: Train index
def train(embeddings: np.ndarray, index_kwargs: Dict[str, Any], index_dir: str):
    index_kwargs.update(
        tempdir=index_dir,
        use_gpu=config.INDEX_USE_GPU,
        train_on_gpu=config.INDEX_TRAIN_ON_GPU,
    )

    index = InteractiveIndex(**index_kwargs)
    index.train(embeddings)


# Step 3: Call webhook to indicate completion
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def notify(url: str, payload: Dict[str, str]):
    r = requests.put(url, json=payload)
    r.raise_for_status()


@dataclass
class TrainingJob:
    path_tmpls: List[
        str
    ]  # Format strings (param: reduction) for paths to saved embedding dictionaries
    index_kwargs: Dict[str, Any]  # Index configuration

    index_id: str  # Index build job identifier
    index_name: str  # Unique index identifier within job
    url: str  # Webhook to PUT to after completion

    sample_rate: float  # Fraction of saved embeddings to randomly sample for training
    reduction: Optional[str]  # Type of embeddings we should use (e.g., average pooled)

    _done: bool = False
    _done_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def finish(self, success: bool, **kwargs):
        with self._done_lock:
            if self._done:
                return
            self._done = True

        notify(
            self.url,
            {
                "index_id": self.index_id,
                "index_name": self.index_name,
                "success": success,
                **kwargs,
            },
        )

    @property
    def done(self):
        with self._done_lock:
            return self._done

    def run(self):
        # TODO(mihirg): Figure out how to handle errors like OOMs and CUDA
        # errors maybe start a subprocess?
        try:
            start_time = time.perf_counter()
            paths = (path_tmpl.format(self.reduction)
                     for path_tmpl in self.path_tmpls)
            embeddings, num_paths_read = load(paths, self.sample_rate)

            train_start_time = time.perf_counter()
            index_dir = config.INDEX_DIR_TMPL.format(
                self.index_id, self.index_name)
            train(embeddings, self.index_kwargs, index_dir)

            end_time = time.perf_counter()
        except Exception as e:
            traceback.print_exc()
            self.finish(False, reason=str(e))
        else:
            self.finish(
                True,
                index_dir=index_dir,
                num_paths_read=num_paths_read,
                profiling=dict(
                    load_time=train_start_time - start_time,
                    train_time=end_time - train_start_time,
                ),
            )


current_job: Optional[TrainingJob] = None
app = Flask(__name__)


@app.route("/", methods=["POST"])
def start():
    global current_job
    if current_job and not current_job.done:
        abort(503, description="Busy")

    payload = request.json or {}
    current_job = TrainingJob(**payload)
    current_job.start()
    return "Started"


def gracefully_shutdown(signum, frame):
    if current_job:
        current_job.finish(False, reason="Preempted")


signal.signal(signal.SIGTERM, gracefully_shutdown)
