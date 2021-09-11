import asyncio
import base64
import functools
import logging
import math
import operator
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import fastcluster
import numpy as np
import sanic.response as resp
from dataclasses_json import dataclass_json
from forager_knn import utils
from sanic import Sanic
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.metrics import precision_score, recall_score

import forager_embedding_server.log
import forager_embedding_server.models as models
from forager_embedding_server.ais import ais_singleiter, get_fscore
from forager_embedding_server.config import CONFIG
from forager_embedding_server.embedding_jobs import EmbeddingInferenceJob
from forager_embedding_server.jobs_data import (ImageList, load_embedding_set,
                                                load_score_set)
from forager_embedding_server.utils import CleanupDict, sha_encode

BUILD_WITH_KUBE = False

if BUILD_WITH_KUBE:
    from forager_knn.clusters import TerraformModule

    from forager_embedding_server.bgsplit_jobs import (BGSplitInferenceJob,
                                                       BGSplitTrainingJob,
                                                       Trainer)

# Create a logger for the server

forager_embedding_server.log.init_logging()
logger = logging.getLogger(__name__)


# GLOBALS

# Start web server
app = Sanic(__name__, log_config=forager_embedding_server.log.LOGGING)
app.update_config({"RESPONSE_TIMEOUT": CONFIG.SANIC_RESPONSE_TIMEOUT})


BUILTIN_MODELS = {
    "clip": models.CLIP(),
    "resnet": models.ResNet(),
}

#
# INDEX
#

FORAGER_EMBEDDINGS_DIR = Path("~/.forager/embeddings").expanduser()
FORAGER_SCORES_DIR = Path("~/.forager/scores").expanduser()
FORAGER_IMAGE_LISTS_DIR = Path("~/.forager/image_lists").expanduser()

current_embedding_jobs: CleanupDict[str, EmbeddingInferenceJob] = CleanupDict(
    lambda job: job.stop()
)


@app.route("/start_embedding_job", methods=["POST"])
async def start_embedding_job(request):
    """
    request["model_output_name"] - name of the model output
    request["splits_to_image_paths"] = {"split_name": ["list of image paths"]}
    request["embedding_type"] - one of ['resnet', 'clip']"""

    model_output_name = request.json["model_output_name"]
    splits_to_image_paths = request.json["splits_to_image_paths"]
    embedding_type = request.json["embedding_type"]

    model_uuid = str(uuid.uuid4())

    FORAGER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FORAGER_IMAGE_LISTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = FORAGER_EMBEDDINGS_DIR / model_uuid
    image_lists_path = FORAGER_IMAGE_LISTS_DIR / model_uuid

    # Create image list
    with open(image_lists_path, "w") as f:
        ImageList.write_from_image_paths(splits_to_image_paths, f)

    # Create embedding job
    job_id = str(uuid.uuid4())
    embedding_job = EmbeddingInferenceJob(
        job_id=job_id,
        image_list_path=str(image_lists_path),
        embedding_type=embedding_type,
        output_path=str(embeddings_path),
    )

    # Run job
    embedding_job.start()

    current_embedding_jobs[job_id] = embedding_job

    return resp.json({"job_id": job_id})


@app.route("/embedding_job_status", methods=["GET"])
async def embedding_job_status(request):
    job_id = request.args["job_id"][0]
    job = current_embedding_jobs.get(job_id)

    status = {
        "has_job": job is not None,
        "finished": job.status["finished"] if job else False,
        "failed": job.status["failed"] if job else False,
        "failure_reason": job.status["failure_reason"] if job else "",
        "status": job.status if job else None,
        "image_list_path": job.image_list_path if job else "",
        "embeddings_path": job.embeddings_path if job else "",
    }
    return resp.json(status)


#
# CLUSTER
#


async def _start_cluster(cluster):
    # Create cluster
    # Hack(mihirg): just attach mounting-related attributes to the cluster object
    cluster.mounted = asyncio.Event()
    await cluster.apply()

    # Mount NFS
    cluster.mount_parent_dir = CONFIG.CLUSTER.MOUNT_DIR / cluster.id
    cluster.mount_parent_dir.mkdir(parents=True, exist_ok=False)

    cluster.mount_dir = cluster.mount_parent_dir / cluster.output[
        "nfs_mount_dir"
    ].lstrip(os.sep)
    cluster.mount_dir.mkdir()

    proc = await asyncio.create_subprocess_exec(
        "sudo",
        "mount",
        cluster.output["nfs_url"],
        str(cluster.mount_dir),
    )
    await proc.wait()
    cluster.mounted.set()


async def _stop_cluster(cluster):
    await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())

    # Unmount NFS
    proc = await asyncio.create_subprocess_exec(
        "sudo", "umount", "-f", "-l", str(cluster.mount_dir)
    )
    await proc.wait()
    try:
        shutil.rmtree(cluster.mount_parent_dir)
    except Exception:
        pass

    # Destroy cluster
    if not CONFIG.CLUSTER.REUSE_EXISTING:
        await cluster.destroy()


# TODO(mihirg): Automatically clean up inactive clusters
if BUILD_WITH_KUBE:
    current_clusters: CleanupDict[str, TerraformModule] = CleanupDict(
        _stop_cluster, app.add_task, CONFIG.CLUSTER.CLEANUP_TIME
    )
else:
    current_clusters = {}


@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    if not BUILD_WITH_KUBE:
        return resp.json({"success": False}, status=400)

    cluster = TerraformModule(
        CONFIG.CLUSTER.TERRAFORM_MODULE_PATH, copy=not CONFIG.CLUSTER.REUSE_EXISTING
    )
    app.add_task(_start_cluster(cluster))
    cluster_id = cluster.id
    current_clusters[cluster_id] = cluster
    return resp.json({"cluster_id": cluster_id})


@app.route("/cluster_status", methods=["GET"])
async def cluster_status(request):
    cluster_id = request.args["cluster_id"][0]
    cluster = current_clusters.get(cluster_id)
    has_cluster = cluster is not None

    status = {
        "has_cluster": has_cluster,
        "ready": has_cluster and cluster.ready.is_set(),
    }
    return resp.json(status)


@app.route("/stop_cluster", methods=["POST"])
async def stop_cluster(request):
    cluster_id = request.json["cluster_id"]
    app.add_task(current_clusters.cleanup_key(cluster_id))
    return resp.text("", status=204)


#
# BACKGROUND SPLITTING
# Note(mihirg): These functions are out of date as of 6/25 and need to be fixed/removed.
#

if BUILD_WITH_KUBE:
    current_models: CleanupDict[str, BGSplitTrainingJob] = CleanupDict(
        lambda job: job.stop()
    )
else:
    current_models = {}


@app.route("/start_bgsplit_job", methods=["POST"])
async def start_bgsplit_job(request):
    logger.info(f"Train request received")
    # HACK(mihirg): sometimes we get empty identifiers (i.e., "") from the server that
    # would otherwise cause a crash here; we should probably figure out why this is, but
    # just filtering out for now.
    pos_identifiers = list(filter(bool, request.json["pos_identifiers"]))
    neg_identifiers = list(filter(bool, request.json["neg_identifiers"]))
    val_pos_identifiers = list(filter(bool, request.json["val_pos_identifiers"]))
    val_neg_identifiers = list(filter(bool, request.json["val_neg_identifiers"]))
    bucket = request.json["bucket"]
    model_name = request.json["model_name"]
    cluster_id = request.json["cluster_id"]
    index_id = request.json["index_id"]
    model_kwargs = request.json["model_kwargs"]
    aux_labels_type = model_kwargs["aux_labels_type"]
    resume_from = request.json["resume_from"]
    pref_worker_id = request.json.get("preferred_worker_id", None)

    restrict_aux_labels = model_kwargs.get("restrict_aux_labels", True)

    # Get cluster
    if cluster_id not in current_clusters:
        return resp.json(
            {"reason": f"Cluster {cluster_id} does not exist."}, status=400
        )

    cluster = current_clusters[cluster_id]
    # lock_id = current_clusters.lock(cluster_id)
    # cluster_unlock_fn = functools.partial(current_clusters.unlock, cluster_id, lock_id)
    await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())

    # Get index
    index = await get_index(index_id)

    # Get image paths from index
    gcs_root_path = os.path.join(CONFIG.GCS_PUBLIC_ROOT_URL, bucket)
    pos_paths = [
        os.path.join(gcs_root_path, index.labels[index.train_identifiers[i]])
        for i in pos_identifiers
    ]
    if len(pos_paths) == 0 and restrict_aux_labels:
        return resp.json(
            {
                "reason": "Can not train model with 0 positives and restricted aux labels."
            },
            status=400,
        )

    neg_paths = [
        os.path.join(gcs_root_path, index.labels[index.train_identifiers[i]])
        for i in neg_identifiers
    ]

    val_pos_paths = [
        os.path.join(gcs_root_path, index.labels[index.train_identifiers[i]])
        for i in val_pos_identifiers
    ]

    val_neg_paths = [
        os.path.join(gcs_root_path, index.labels[index.train_identifiers[i]])
        for i in val_neg_identifiers
    ]

    unused_identifiers = (
        set(index.get_train_identifiers())
        .difference(set(pos_identifiers))
        .difference(set(neg_identifiers))
        .difference(set(val_pos_identifiers))
        .difference(set(val_neg_identifiers))
    )

    if len(neg_paths) == 0 and restrict_aux_labels:
        return resp.json(
            {
                "reason": (
                    "Can not train model with 0 negatives and " "restricted aux labels."
                )
            },
            status=400,
        )

    unlabeled_paths = [
        os.path.join(gcs_root_path, index.labels[index.train_identifiers[i]])
        for i in list(unused_identifiers)
    ]

    http_session = utils.create_unlimited_aiohttp_session()
    # 1. If aux labels have not been generated, then generate them
    # TODO(fpoms): Actually generate aux labels; and maybe move this to index build?
    alt = aux_labels_type
    aux_labels_gcs_path = CONFIG.AUX.GCS_PUBLIC_TMPL.format(index_id, alt)

    # 2. Train BG Split model
    trainers = [Trainer(url) for url in cluster.output["bgsplit_trainer_urls"]]
    model_id = str(uuid.uuid4())

    preferred_trainer = None
    if pref_worker_id:
        for trainer in trainers:
            if trainer.trainer_id == pref_worker_id:
                preferred_trainer = trainer
                break

    training_job = BGSplitTrainingJob(
        pos_paths=pos_paths,
        neg_paths=neg_paths,
        val_pos_paths=val_pos_paths,
        val_neg_paths=val_neg_paths,
        unlabeled_paths=unlabeled_paths,
        user_model_kwargs=model_kwargs,
        aux_labels_path=aux_labels_gcs_path,
        model_name=model_name,
        model_id=model_id,
        resume_from=resume_from,
        trainers=trainers,
        preferred_trainer=preferred_trainer,
        cluster=cluster,
        session=http_session,
    )
    current_models[model_id] = training_job
    training_job.start()
    logger.info(
        f"Train ({training_job.model_name}): started with "
        f"{len(pos_paths)} positives, {len(neg_paths)} negatives, and "
        f"{len(unlabeled_paths)} unlabeled examples."
    )

    return resp.json({"model_id": model_id})


@app.route(CONFIG.BGSPLIT.TRAINER_STATUS_ENDPOINT, methods=["PUT"])
async def bgsplit_training_status(request):
    model_id = request.json["model_id"]
    if model_id in current_models:
        await current_models[model_id].handle_result(request.json)
    return resp.text("", status=204)


@app.route("/bgsplit_job_status", methods=["GET"])
async def bgsplit_job_status(request):
    model_id = request.args["model_id"][0]
    if model_id in current_models:
        model = current_models[model_id]
        status = model.status
        status["has_model"] = status["finished"] and not status["failed"]
        status["checkpoint_path"] = model.model_checkpoint
    else:
        status = {"has_model": False, "failed": False}
    return resp.json(status)


if BUILD_WITH_KUBE:
    current_model_inference_jobs: CleanupDict[str, BGSplitInferenceJob] = CleanupDict(
        lambda job: job.stop()
    )
else:
    current_model_inference_jobs = {}


@app.route("/start_bgsplit_inference_job", methods=["POST"])
async def start_bgsplit_inference_job(request):
    # HACK(mihirg): sometimes we get empty identifiers (i.e., "") from the server that
    # would otherwise cause a crash here; we should probably figure out why this is, but
    # just filtering out for now.
    bucket = request.json["bucket"]
    model_id = request.json["model_id"]
    checkpoint_path = request.json["checkpoint_path"]
    cluster_id = request.json["cluster_id"]
    index_id = request.json["index_id"]

    # Get cluster
    cluster = current_clusters[cluster_id]
    # lock_id = current_clusters.lock(cluster_id)
    # cluster_unlock_fn = functools.partial(current_clusters.unlock, cluster_id, lock_id)
    await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())

    # Get index
    index = await get_index(index_id)

    # Get image paths from index
    gcs_root_path = os.path.join(CONFIG.GCS_PUBLIC_ROOT_URL, bucket)
    all_paths = index.labels

    http_session = utils.create_unlimited_aiohttp_session()
    job_id = str(uuid.uuid4())

    inference_job = BGSplitInferenceJob(
        job_id=job_id,
        paths=all_paths,
        bucket=bucket,
        model_id=model_id,
        model_checkpoint_path=checkpoint_path,
        cluster=cluster,
        session=http_session,
    )
    current_model_inference_jobs[job_id] = inference_job
    inference_job.start()
    logger.info(f"Inference ({checkpoint_path}): started ")

    return resp.json({"job_id": job_id})


@app.route("/bgsplit_inference_job_status", methods=["GET"])
async def bgsplit_inference_job_status(request):
    job_id = request.args["job_id"][0]
    if job_id in current_model_inference_jobs:
        job = current_model_inference_jobs[job_id]
        status = job.status
        status["has_output"] = status["finished"]
    else:
        status = {"has_output": False, "finished": False}
    return resp.json(status)


# TODO(fpoms): fix this
# @app.route("/stop_bgsplit_inference_job", methods=["POST"])
# async def bgsplit_inference_job_status(request):
#     job_id = request.args["job_id"][0]
#     if job_id in current_model_inference_jobs:
#         job = current_model_inference_jobs[job_id]
#         job.stop()
#         del current_model_inference_jobs[job_id]
#         status["has_output"] = False
#         status["finished"] = True
#         status["failed"] = False
#         return resp.json(status)
#     else:
#         status = {"reason": f"Job id {job_id} does not exist."}
#         return resp.json(status, status=400)


#
# QUERYING
#


@app.route("/perform_clustering", methods=["POST"])
async def perform_clustering(request):
    identifiers = request.json["identifiers"]
    image_list_path = request.json["image_list_path"]
    embedding_set_path = request.json["embedding_set_path"]

    embedding_set = load_embedding_set(embedding_set_path, image_list_path)
    clustering = embedding_set.cluster_identifiers(identifiers)
    return resp.json({"clustering": clustering})


@app.route("generate_embedding", methods=["POST"])
async def generate_embedding(request):
    identifier = request.json.get("identifier")
    if identifier:
        image_list_path = request.json["image_list_path"]
        embedding_set_path = request.json["embedding_set_path"]
        embedding_set = load_embedding_set(embedding_set_path, image_list_path)
        embedding = embedding_set.get_embeddings([identifier])[0]
    else:
        model = request.json["model"]
        image_data = re.sub("^data:image/.+;base64,", "", request.json["image_data"])
        embedding = await utils.run_in_executor(
            BUILTIN_MODELS[model].embed_image_bytes,
            base64.b64decode(image_data),
        )
    return resp.json({"embedding": utils.numpy_to_base64(embedding)})


@app.route("generate_text_embedding", methods=["POST"])
async def generate_text_embedding(request):
    text = request.json["text"]
    model = request.json["model"]
    embedding = await utils.run_in_executor(BUILTIN_MODELS[model].embed_text, text)
    return resp.json({"embedding": utils.numpy_to_base64(embedding)})


@app.route("/query_knn", methods=["POST"])
async def query_knn(request):
    embeddings = request.json["embeddings"]
    use_dot_product = request.json["use_dot_product"]
    use_full_image = request.json["use_full_image"]
    image_list_path = request.json["image_list_path"]
    embedding_set_path = request.json["embedding_set_path"]

    assert use_full_image, "Spaital queries not supported yet"

    embedding_set = load_embedding_set(embedding_set_path, image_list_path)
    query_vector = np.mean([utils.base64_to_numpy(e) for e in embeddings], axis=0)
    query_results = embedding_set.query_brute_force(
        query_vector, dot_product=use_dot_product
    )
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/train_svm", methods=["POST"])
async def train_svm(request):
    # HACK(mihirg): sometimes we get empty identifiers (i.e., "") from the server that
    # would otherwise cause a crash here; we should probably figure out why this is, but
    # just filtering out for now.
    pos_identifiers: List[str] = list(filter(bool, request.json["pos_identifiers"]))
    neg_identifiers: List[str] = list(filter(bool, request.json["neg_identifiers"]))
    image_list_path = request.json["image_list_path"]
    embedding_set_path = request.json["embedding_set_path"]

    embedding_set = load_embedding_set(embedding_set_path, image_list_path)

    # Get positive and negative image embeddings from local flat index
    pos_vectors = embedding_set.get_embeddings(pos_identifiers)
    neg_vectors = embedding_set.get_embeddings(neg_identifiers)
    assert len(pos_vectors) > 0 and len(neg_vectors) > 0

    # Train SVM and return serialized vector
    training_features = np.concatenate((pos_vectors, neg_vectors))
    training_labels = np.array([1] * len(pos_vectors) + [0] * len(neg_vectors))
    model = svm.LinearSVC(C=0.1)
    model.fit(training_features, training_labels)

    w = np.array(model.coef_[0] * 1000, dtype=np.float32)
    predicted = model.predict(training_features)
    precision = precision_score(training_labels, predicted)
    recall = recall_score(training_labels, predicted)

    return resp.json(
        {
            "svm_vector": utils.numpy_to_base64(w),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
            "num_positives": len(pos_vectors),
            "num_negatives": len(neg_vectors),
        }
    )


@app.route("/query_svm", methods=["POST"])
async def query_svm(request):
    score_min = float(request.json["score_min"])
    score_max = float(request.json["score_max"])
    svm_vector = utils.base64_to_numpy(request.json["svm_vector"])
    image_list_path = request.json["image_list_path"]
    embedding_set_path = request.json["embedding_set_path"]

    embedding_set = load_embedding_set(embedding_set_path, image_list_path)
    query_results = embedding_set.query_brute_force(
        svm_vector, dot_product=True, min_d=score_min, max_d=score_max
    )
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/query_ranking", methods=["POST"])
async def query_ranking(request):
    score_min = float(request.json["score_min"])
    score_max = float(request.json["score_max"])
    image_list_path = request.json["image_list_path"]
    score_set_path = request.json["score_set_path"]

    score_set = load_score_set(score_set_path, image_list_path)
    query_results = score_set.rank_brute_force(score_min, score_max)
    return resp.json({"results": [r.to_dict() for r in query_results]})


#
# VALIDATION
#


@app.route("/query_metrics", methods=["POST"])
async def query_metrics(request):
    include_fp_fn = request.json.get("include_fp_fn", False)  # type: bool
    identifiers = request.json["identifiers"]  # type: List[str]
    labels = request.json["labels"]  # type: List[bool]
    weights = request.json["weights"]  # type: List[float]
    image_list_path = request.json["image_list_path"]  # type: str
    score_set_path = request.json["score_set_path"]  # type: str

    assert len(identifiers) == len(labels) == len(weights)
    num_labeled = len(identifiers)

    score_set = load_score_set(score_set_path, image_list_path)

    prob_pos = score_set.get_scores(identifiers)
    y_pred = prob_pos > CONFIG.DNN_SCORE_CLASSIFICATION_THRESHOLD
    y_test = np.array(labels)
    rows = np.arange(num_labeled)
    weights = np.array(weights)

    precision, precision_std, _ = get_fscore(y_pred, y_test, rows, weights * y_pred)
    recall, recall_std, _ = get_fscore(y_pred, y_test, rows, weights * y_test)
    f1, f1_std, _ = get_fscore(
        y_pred, y_test, rows, weights * (0.5 * y_pred + 0.5 * y_test)
    )

    false_positives = []
    false_negatives = []
    if include_fp_fn:
        for identifier, label, score in zip(identifiers, labels, prob_pos):
            result = QueryResult(
                score_set.images.get_ind(identifier), float(score), identifier
            )
            if score > CONFIG.DNN_SCORE_CLASSIFICATION_THRESHOLD and not label:
                false_positives.append(result)
            elif score <= CONFIG.DNN_SCORE_CLASSIFICATION_THRESHOLD and label:
                false_negatives.append(result)

    false_positives.sort(key=operator.attrgetter("dist"), reverse=True)  # descending
    false_negatives.sort(key=operator.attrgetter("dist"))  # ascending

    results = {
        "precision": precision,
        "precision_std": precision_std,
        "recall": recall,
        "recall_std": recall_std,
        "f1": f1,
        "f1_std": f1_std,
        "false_positives": [r.to_dict() for r in false_positives],
        "false_negatives": [r.to_dict() for r in false_negatives],
        "num_false_positives": len(false_positives),
        "num_false_negatives": len(false_negatives),
        "num_labeled": num_labeled,
    }
    for metric in ("precision", "recall", "f1"):
        for k in (metric, f"{metric}_std"):
            if math.isnan(results[k]):
                results[k] = None
    return resp.json(results)


@app.route("/query_active_validation", methods=["POST"])
async def query_active_validation(request):
    identifiers = request.json["identifiers"]  # type: List[str]
    labels = request.json["labels"]  # type: List[bool]
    current_f1 = float(request.json["current_f1"])
    image_list_path = request.json["image_list_path"]
    score_set_path = request.json["score_set_path"]

    assert len(identifiers) == len(labels)

    score_set = load_score_set(score_set_path, image_list_path)

    all_val_inds = score_set.images.get_inds_for_split("val")
    all_val_identifiers = score_set.images.get_identifiers(all_val_inds)
    prob_pos = score_set.get_scores(inds=all_val_inds)
    y_pred = prob_pos > CONFIG.DNN_SCORE_CLASSIFICATION_THRESHOLD
    sample_budget = max(2 * len(labels), CONFIG.ACTIVE_VAL_STARTING_BUDGET)
    g = current_f1
    alpha = 0.5

    val_identifiers_to_inds_in_val = {
        identifier: i for i, identifier in enumerate(all_val_identifiers)
    }
    known_rows_inds = list(map(val_identifiers_to_inds_in_val.get, identifiers))
    known_rows = np.zeros(len(y_pred), dtype=bool)
    known_rows[known_rows_inds] = True

    y_test = np.zeros(len(y_pred), dtype=bool)
    y_test[known_rows_inds] = labels

    # Restrict sampling domain in early iterations when there aren't many
    # labeled positives
    i = np.log2(sample_budget / 10)  # inverse of sample_budget = 10 * (2 ** i)
    poses = y_pred.sum()
    t = int(3 * (i + 1) * poses)
    if t < len(y_pred):
        filter_rows = np.argpartition(prob_pos, -t)[-t:]
    else:
        filter_rows = np.arange(len(y_pred))

    # Use AIS algorithm to sample rows to label
    rows, weights = ais_singleiter(
        y_pred=y_pred,
        y_test=y_test[known_rows],
        prob_pos=prob_pos,
        sample_budget=sample_budget,
        g=g,
        alpha=alpha,
        known_rows=known_rows,
        filter_rows=filter_rows,
    )
    rows = filter_rows[rows]
    weights *= len(rows) / len(y_pred)

    old_identifiers = set(identifiers)
    new_identifiers = []
    identifiers_to_weights: Dict[str, float] = {}
    for ind_in_val, weight in zip(rows, weights):
        ind = all_val_inds[ind_in_val]
        identifier = score_set.images.get_identifier(ind)
        identifiers_to_weights[identifier] = weight
        if identifier not in old_identifiers:
            new_identifiers.append(identifier)

    return resp.json(
        {
            "identifiers": new_identifiers,
            "weights": identifiers_to_weights,
        }
    )


# CLEANUP


@app.listener("after_server_stop")
async def cleanup(app, loop):
    print("Terminating:")
    await _cleanup_clusters()


@utils.log_exception_from_coro_but_return_none
async def _cleanup_clusters():
    if BUILD_WITH_KUBE:
        n = len(current_clusters)
        await current_clusters.clear_async()
        print(f"- killed {n} clusters")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
