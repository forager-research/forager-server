import functools
import threading
import backoff
import requests
import pickle
import time
import traceback
import logging
import os.path
import json
import sys
import shutil
import types
import numpy as np
import tempfile
import weakref
from flask import Flask, request, abort
from werkzeug.exceptions import HTTPException
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

import config
from training_loop import TrainingLoop
from util import download

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


# For caching dataset images
data_cache_dir = tempfile.TemporaryDirectory()
data_finalizer = weakref.finalize(
    data_cache_dir, shutil.rmtree, data_cache_dir.name)


# Step 3: Call webhook to indicate completion
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def notify(url: str, payload: Dict[str, Any]):
    r = requests.put(url, data=json.dumps(payload))
    r.raise_for_status()


@dataclass
class TrainingJob:
    train_positive_paths: List[str]
    train_negative_paths: List[str]
    train_unlabeled_paths: List[str]
    val_positive_paths: List[str]
    val_negative_paths: List[str]
    val_unlabeled_paths: List[str]
    model_kwargs: Dict[str, Any]

    model_id: str
    model_name: str
    notify_url: str

    _lock: threading.Lock

    _done: bool = False
    _done_lock: threading.Lock = field(default_factory=threading.Lock)

    last_checkpoint_path: Optional[str] = None
    training_loop: Optional[TrainingLoop] = None
    model_suffix: Optional[str] = None

    def update_model_id_and_paths(self, payload):
        # NOTE(fpoms): We don't currently allow changing the model_kwargs
        self.train_positive_paths = payload['train_positive_paths']
        self.train_negative_paths = payload['train_negative_paths']
        self.train_unlabeled_paths = payload['train_unlabeled_paths']
        self.val_positive_paths = payload['val_positive_paths']
        self.val_negative_paths = payload['val_negative_paths']
        self.val_unlabeled_paths = payload['val_unlabeled_paths']
        self.model_id = payload['model_id']
        self.model_name = payload['model_name']
        with self._done_lock:
            self._done = False

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def notify_status(
            self, success: bool=False, failed: bool=False, **kwargs):
        data={
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_suffix": self.model_suffix,
            "success": success,
            "failed": failed,
            **kwargs,
        }
        logger.debug(f'Sending notify: {data}')
        notify(self.notify_url, data)

    def finish(self, success: bool, failed: bool=False, **kwargs):
        with self._done_lock:
            if self._done:
                return
            self._done = True

        self.notify_status(success=success, failed=failed, **kwargs)
        self._lock.release()

    @property
    def done(self):
        with self._done_lock:
            return self._done

    def run(self):
        # TODO(mihirg): Figure out how to handle errors like OOMs and CUDA errors,
        # maybe start a subprocess?
        try:
            start_time = time.perf_counter()
            if self.training_loop:
                logger.info('Reusing training model')
                train_start_time = time.perf_counter()
                self.training_loop.setup_resume(
                    train_positive_paths=self.train_positive_paths,
                    train_negative_paths=self.train_negative_paths,
                    train_unlabeled_paths=self.train_unlabeled_paths,
                    val_positive_paths=self.val_positive_paths,
                    val_negative_paths=self.val_negative_paths,
                    val_unlabeled_paths=self.val_unlabeled_paths,
                )
            else:
                aux_labels_path = self.model_kwargs['aux_labels_path']
                logger.info(f'Downloading aux labels: {aux_labels_path}')
                aux_labels = {}
                data = download(aux_labels_path)
                auxiliary_labels = pickle.loads(data)
                for p, v in auxiliary_labels.items():
                    aux_labels[os.path.basename(p)] = v
                self.model_kwargs['aux_labels'] = aux_labels

                model_dir = config.MODEL_DIR_TMPL.format(
                    self.model_name, self.model_id
                )
                self.model_kwargs['model_dir'] = model_dir

                self.model_suffix = config.MODEL_SUFFIX.format(
                    self.model_name, self.model_id)

                log_dir = config.LOG_DIR_TMPL.format(
                    self.model_name, self.model_id
                )
                self.model_kwargs['log_dir'] = log_dir

                end_time = time.perf_counter()
                train_start_time = time.perf_counter()
                # Train
                logger.info('Creating training model')
                self.training_loop = TrainingLoop(
                    model_kwargs=self.model_kwargs,
                    train_positive_paths=self.train_positive_paths,
                    train_negative_paths=self.train_negative_paths,
                    train_unlabeled_paths=self.train_unlabeled_paths,
                    val_positive_paths=self.val_positive_paths,
                    val_negative_paths=self.val_negative_paths,
                    val_unlabeled_paths=self.val_unlabeled_paths,
                    data_cache_dir=data_cache_dir.name,
                    notify_callback=self.notify_status,
                )
            logger.info('Running training')
            # Notify to send the model suffix so we can show the tensorboard url
            # asap
            self.notify_status()
            self.last_checkpoint_path = self.training_loop.run()
            end_time = time.perf_counter()
        except Exception as e:
            logger.exception(f'Exception: {traceback.print_exc()}')
            self.finish(False, failed=True, reason=str(e))
        else:
            logger.info('Finished training')
            self.finish(
                True,
                model_dir=self.training_loop.model_dir,
                model_checkpoint=self.last_checkpoint_path,
                profiling=dict(
                    load_time=train_start_time - start_time,
                    train_time=end_time - train_start_time,
                ),
                debug=dict(
                    train_epoch_loss=self.training_loop.train_epoch_loss,
                    train_epoch_main_loss=self.training_loop.train_epoch_main_loss,
                    train_epoch_aux_loss=self.training_loop.train_epoch_aux_loss
                )
            )


working_lock = threading.Lock()
app = Flask(__name__)

last_job: Optional[TrainingJob] = None


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@app.route("/", methods=["POST"])
def start():
    global last_job
    try:
        payload = request.json or {}
    except Exception as e:
        abort(400, description=str(e))

    if not working_lock.acquire(blocking=False):
        abort(503, description="Busy")

    logger.debug(f'Received request')
    payload["_lock"] = working_lock
    log_payload = dict(payload)
    for k in ['train_positive_paths', 'train_negative_paths', 'train_unlabeled_paths',
              'val_positive_paths', 'val_negative_paths', 'val_unlabeled_paths']:
        log_payload[k] = len(log_payload[k])
    logger.debug(f'Received job payload: {log_payload}')
    resume_from_checkpoint = payload['model_kwargs']['resume_from']
    if resume_from_checkpoint and last_job and \
       payload['model_kwargs'].get('resume_training', False) and \
       resume_from_checkpoint == last_job.last_checkpoint_path:
        logger.info(f'Resuming from prior job ({last_job.model_id}) for model {payload["model_id"]}')
        current_job = last_job
        current_job.update_model_id_and_paths(payload)
    else:
        last_job = None
        logger.info(f'Starting new job for model {payload["model_id"]}')
        current_job = TrainingJob(**payload)
    current_job.start()
    logger.debug(f'Started job')
    last_job = current_job
    return "Started"
