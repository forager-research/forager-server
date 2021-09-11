import distutils.util
import functools
import itertools
import json
import logging
import math
import operator
import os
import random
import shutil
import time
import uuid
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import requests
from django.conf import settings
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_list_or_404, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from expiringdict import ExpiringDict
from rest_framework.decorators import api_view

from .models import (Annotation, Category, CategoryCount, Dataset, DatasetItem,
                     DNNModel, Mode, ModelOutput, User)

BUILTIN_MODES = ["POSITIVE", "NEGATIVE", "HARD_NEGATIVE", "UNSURE"]

FORAGER_EMBEDDINGS_DIR = Path("~/.forager/embeddings").expanduser()
FORAGER_SCORES_DIR = Path("~/.forager/scores").expanduser()
FORAGER_IMAGE_LISTS_DIR = Path("~/.forager/image_lists").expanduser()

logger = logging.getLogger(__name__)

#
# CLUSTER
#


@api_view(["POST"])
@csrf_exempt
def start_cluster(request):
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_cluster",
    )
    response_data = r.json()
    return JsonResponse(
        {
            "status": "success",
            "cluster_id": response_data["cluster_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_cluster_status(request, cluster_id):
    params = {"cluster_id": cluster_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/cluster_status", params=params
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def stop_cluster(request, cluster_id):
    params = {"cluster_id": cluster_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/stop_cluster",
        json=params,
    )
    return JsonResponse(
        {
            "status": "success",
        }
    )


#
# MODEL TRAINING
# Note(mihirg): These functions are out of date as of 6/25 and may need to be
# fixed/removed, along with their counterparts on the index server.
#


@api_view(["POST"])
@csrf_exempt
def create_model(request, dataset_name, dataset=None):
    payload = json.loads(request.body)
    model_name = payload["model_name"]
    cluster_id = payload["cluster_id"]
    bucket_name = payload["bucket"]
    index_id = payload["index_id"]
    pos_tags = parse_tag_set_from_query(payload["pos_tags"])
    neg_tags = parse_tag_set_from_query(payload["neg_tags"])
    val_pos_tags = parse_tag_set_from_query(payload["val_pos_tags"])
    val_neg_tags = parse_tag_set_from_query(payload["val_neg_tags"])
    augment_negs = bool(payload["augment_negs"])
    model_kwargs = payload["model_kwargs"]
    resume_model_id = payload.get("resume", None)

    dataset = get_object_or_404(Dataset, name=dataset_name)
    eligible_images = DatasetItem.objects.filter(dataset=dataset, is_val=False)
    categories = Category.objects.filter(
        tag_sets_to_query(pos_tags, neg_tags, val_pos_tags, val_neg_tags)
    )
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_images,
        category__in=categories,
    )
    tags_by_pk = get_tags_from_annotations(annotations)

    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    val_pos_dataset_item_pks = []
    val_neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags for t in tags):
            neg_dataset_item_pks.append(pk)
        elif any(t in val_pos_tags for t in tags):
            val_pos_dataset_item_pks.append(pk)
        elif any(t in val_neg_tags for t in tags):
            val_neg_dataset_item_pks.append(pk)

    # Augment with randomly sampled negatives if requested
    num_extra_negs = settings.BGSPLIT_NUM_NEGS_MULTIPLIER * len(
        pos_dataset_item_pks
    ) - len(neg_dataset_item_pks)
    if augment_negs and num_extra_negs > 0:
        # Uses "include" and "exclude" category sets from request
        all_eligible_pks = filtered_images(
            request,
            dataset,
            exclude_pks=(
                pos_dataset_item_pks
                + neg_dataset_item_pks
                + val_pos_dataset_item_pks
                + val_neg_dataset_item_pks
            ),
        )
        sampled_pks = random.sample(
            all_eligible_pks, min(len(all_eligible_pks), num_extra_negs)
        )
        neg_dataset_item_pks.extend(sampled_pks)

    pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    val_pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=val_pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    val_neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=val_neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )

    if resume_model_id:
        resume_model = get_object_or_404(DNNModel, model_id=resume_model_id)
        resume_model_path = resume_model.checkpoint_path
    else:
        resume_model = None
        resume_model_path = None

    params = {
        "pos_identifiers": pos_dataset_item_internal_identifiers,
        "neg_identifiers": neg_dataset_item_internal_identifiers,
        "val_pos_identifiers": val_pos_dataset_item_internal_identifiers,
        "val_neg_identifiers": val_neg_dataset_item_internal_identifiers,
        "augment_negs": augment_negs,
        "model_kwargs": model_kwargs,
        "model_name": model_name,
        "bucket": bucket_name,
        "cluster_id": cluster_id,
        "index_id": index_id,
        "resume_from": resume_model_path,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_bgsplit_job",
        json=params,
    )
    response_data = r.json()

    if r.status_code != 200:
        return JsonResponse(
            {"status": "failure", "reason": response_data.get("reason", "")},
            status=r.status_code,
        )

    m = DNNModel(
        dataset=dataset,
        name=model_name,
        model_id=response_data["model_id"],
        category_spec={
            "augment_negs": augment_negs,
            "pos_tags": payload["pos_tags"],
            "neg_tags": payload["neg_tags"],
            "augment_negs_include": payload.get("include", []) if augment_negs else [],
            "augment_negs_exclude": payload.get("exclude", []) if augment_negs else [],
        },
    )
    model_epoch = -1 + model_kwargs.get("epochs_to_run", 1)
    if resume_model_id:
        m.resume_model_id = resume_model_id
        if model_kwargs.get("resume_training", False):
            model_epoch += resume_model.epoch + 1
    m.epoch = model_epoch
    m.save()

    return JsonResponse(
        {
            "status": "success",
            "model_id": response_data["model_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_model_status(request, model_id):
    params = {"model_id": model_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/bgsplit_job_status", params=params
    )
    response_data = r.json()
    if response_data["has_model"]:
        # Index has been successfully created & uploaded -> persist
        m = get_object_or_404(DNNModel, model_id=model_id)
        m.checkpoint_path = response_data["checkpoint_path"]
        m.save()

    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def update_model(request):
    payload = json.loads(request.body)
    # user = payload["user"]
    old_model_name = payload["old_model_name"]
    new_model_name = payload["new_model_name"]

    models = get_list_or_404(DNNModel, name=old_model_name)
    for m in models:
        m.name = new_model_name
        m.save()

    return JsonResponse({"success": True})


@api_view(["POST"])
@csrf_exempt
def delete_model(request):
    payload = json.loads(request.body)
    model_name = payload["model_name"]
    # cluster_id = payload['cluster_id']
    models = get_list_or_404(DNNModel, name=model_name)
    for m in models:
        # TODO(fpoms): delete model data stored on NFS?
        # shutil.rmtree(os.path.join(m.checkpoint_path, '..'))
        shutil.rmtree(m.output_directory, ignore_errors=True)
        m.delete()

    return JsonResponse({"success": True})


@api_view(["POST"])
@csrf_exempt
def run_model_inference(request, dataset_name, dataset=None):
    payload = json.loads(request.body)
    model_id = payload["model_id"]
    cluster_id = payload["cluster_id"]
    bucket_name = payload["bucket"]
    index_id = payload["index_id"]

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_checkpoint_path = get_object_or_404(
        DNNModel, model_id=model_id
    ).checkpoint_path
    if model_checkpoint_path is None or len(model_checkpoint_path) == 0:
        return JsonResponse(
            {
                "status": "failure",
                "reason": f"Model {model_id} does not have a model checkpoint.",
            },
            status=400,
        )

    params = {
        "bucket": bucket_name,
        "model_id": model_id,
        "checkpoint_path": model_checkpoint_path,
        "cluster_id": cluster_id,
        "index_id": index_id,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_bgsplit_inference_job",
        json=params,
    )
    response_data = r.json()

    return JsonResponse(
        {
            "status": "success",
            "job_id": response_data["job_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_model_inference_status(request, job_id):
    params = {"job_id": job_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/bgsplit_inference_job_status",
        params=params,
    )
    response_data = r.json()
    if response_data["has_output"]:
        model_id = response_data["model_id"]
        # Index has been successfully created & uploaded -> persist
        m = get_object_or_404(DNNModel, model_id=model_id)
        m.output_directory = response_data["output_dir"]
        m.save()

    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def stop_model_inference(request, job_id):
    params = {"job_id": job_id}
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/stop_bgsplit_inference_job", json=params
    )
    response_data = r.json()
    return JsonResponse(response_data, status=r.status_code)


#
# FRONTEND ENDPOINTS
#


Tag = namedtuple("Tag", "category value")  # type: NamedTuple[str, str]
Box = namedtuple(
    "Box", "category value x1 y1 x2 y2"
)  # type: NamedTuple[str, str, float, float, float, float]
PkType = int


@dataclass
class ResultSet:
    type: str
    ranking: List[PkType]
    distances: List[float]


# TODO(fpoms): this needs to be wrapped in a lock so that
# updates are atomic across concurrent requests
current_result_sets = ExpiringDict(
    max_age_seconds=30 * 60,
    max_len=50,
)  # type: Dict[str, ResultSet]


def parse_tag_set_from_query(s):
    if isinstance(s, list):
        parts = s
    elif isinstance(s, str) and s:
        parts = s.split(",")
    else:
        parts = []
    ts = set()
    for part in parts:
        if not part:
            continue
        category, value_str = part.split(":")
        ts.add(Tag(category, value_str))
    return ts


def tag_sets_to_query(*tagsets):
    merged = set().union(*tagsets)
    if not merged:
        return Q(pk__in=[])

    return Q(
        annotation__in=Annotation.objects.filter(
            functools.reduce(
                operator.or_,
                [Q(category__name=t.category, mode__name=t.value) for t in merged],
            )
        )
    )


def serialize_tag_set_for_client(ts):
    return [{"category": t.category, "value": t.value} for t in sorted(list(ts))]


def serialize_boxes_for_client(bs):
    return [
        {
            "category": b.category,
            "value": b.value,
            "x1": b.x1,
            "y1": b.y1,
            "x2": b.x2,
            "y2": b.y2,
        }
        for b in sorted(list(bs))
    ]


def get_tags_from_annotations(annotations):
    tags_by_pk = defaultdict(list)
    annotations = annotations.filter(is_box=False)
    ann_dicts = annotations.values("dataset_item__pk", "category__name", "mode__name")
    for ann in ann_dicts:
        pk = ann["dataset_item__pk"]
        category = ann["category__name"]
        mode = ann["mode__name"]
        tags_by_pk[pk].append(Tag(category, mode))
    return tags_by_pk


def get_boxes_from_annotations(annotations):
    boxes_by_pk = defaultdict(list)
    annotations = annotations.filter(is_box=True)
    ann_dicts = annotations.values(
        "dataset_item__pk",
        "category__name",
        "mode__name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    )
    for ann in ann_dicts:
        pk = ann["dataset_item__pk"]
        category = ann["category__name"]
        mode = ann["mode__name"]
        box = (ann["bbox_x1"], ann["bbox_y1"], ann["bbox_x2"], ann["bbox_y2"])
        boxes_by_pk[pk].append(Box(category, mode, *box))
    return boxes_by_pk


def filtered_images(request, dataset, exclude_pks=None) -> List[PkType]:
    filt_start = time.time()
    if request.method == "POST":
        payload = json.loads(request.body)
        include_tags = parse_tag_set_from_query(payload.get("include"))
        exclude_tags = parse_tag_set_from_query(payload.get("exclude"))
        pks = [i for i in payload.get("subset", []) if i]
        split = payload.get("split", "train")
        offset_to_return = int(payload.get("offset", 0))
        num_to_return = int(payload.get("num", -1))
    else:
        include_tags = parse_tag_set_from_query(request.GET.get("include"))
        exclude_tags = parse_tag_set_from_query(request.GET.get("exclude"))
        pks = [i for i in request.GET.get("subset", "").split(",") if i]
        split = request.GET.get("split", "train")
        offset_to_return = int(request.GET.get("offset", 0))
        num_to_return = int(request.GET.get("num", -1))

    end_to_return = None if num_to_return == -1 else offset_to_return + num_to_return

    dataset_items = None
    is_val = split == "val"

    db_start = time.time()
    # Get pks for dataset items of interest
    if pks and exclude_pks:
        # Get specific pks - excluded pks if requested
        exclude_pks = set(exclude_pks)
        pks = [pk for pk in pks if pk not in exclude_pks]
    elif not pks:
        # Otherwise get all dataset items - exclude pks
        dataset_items = DatasetItem.objects.filter(dataset=dataset, is_val=is_val)
        if exclude_pks:
            dataset_items = dataset_items.exclude(pk__in=exclude_pks)
        pks = dataset_items.values_list("pk", flat=True)
    db_end = time.time()

    result = None
    db_tag_start = time.time()
    if not include_tags and not exclude_tags:
        # If no tags specified, just return retrieved pks
        result = pks
    else:
        # Otherwise, filter using include and exclude tags
        if dataset_items is None:
            dataset_items = DatasetItem.objects.filter(pk__in=pks)

        if include_tags:
            dataset_items = dataset_items.filter(tag_sets_to_query(include_tags))
        dataset_items = dataset_items.exclude(tag_sets_to_query(exclude_tags))

        result = dataset_items.values_list("pk", flat=True)

    db_tag_end = time.time()
    result = list(result[offset_to_return:end_to_return])
    filt_end = time.time()
    print(
        f"filtered_images: tot: {filt_end-filt_start}, "
        f"db ({len(result)} items): {db_end-db_start}, db tag: {db_tag_end-db_tag_start}"
    )
    return result


def process_image_query_results(request, dataset, query_response):
    filtered_pks = filtered_images(request, dataset)
    # TODO(mihirg): Eliminate this database call by directly returning pks from backend
    dataset_items = DatasetItem.objects.filter(pk__in=filtered_pks)
    dataset_items_by_path = {di.path: di for di in dataset_items}

    distances = []
    ordered_pks = []
    for r in query_response["results"]:
        if r["label"] in dataset_items_by_path:
            ordered_pks.append(dataset_items_by_path[r["label"]].pk)
            distances.append(r["dist"])
    return dict(
        pks=ordered_pks,
        distances=distances,
    )


def create_result_set(results, type):
    pks = results["pks"]
    distances = results["distances"]
    result_set_id = str(uuid.uuid4())
    current_result_sets[result_set_id] = ResultSet(
        type=type, ranking=pks, distances=distances
    )
    return {
        "id": result_set_id,
        "num_results": len(pks),
        "type": type,
    }


@api_view(["GET"])
@csrf_exempt
def get_results(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    clustering_model_output_id = request.GET["clustering_model_output_id"]
    result_set_id = request.GET["result_set_id"]
    offset_to_return = int(request.GET.get("offset", 0))
    num_to_return = int(request.GET.get("num", 500))

    # Retrieve slice of pks from result set
    result_set = current_result_sets[result_set_id]
    pks = result_set.ranking[offset_to_return : offset_to_return + num_to_return]
    distances = result_set.distances[
        offset_to_return : offset_to_return + num_to_return
    ]

    # Query full DatasetItem objects
    dataset_items_by_pk = DatasetItem.objects.in_bulk(pks)
    dataset_items = [dataset_items_by_pk[pk] for pk in pks]  # preserve order

    # Parse paths
    dataset_item_identifiers = []
    internal_identifiers = []
    dataset_item_paths = []
    for di in dataset_items:
        dataset_item_identifiers.append(di.pk)
        internal_identifiers.append(di.identifier)
        directory = dataset.val_directory if di.is_val else dataset.train_directory
        if directory.startswith("gs://"):
            bucket_name = dataset.train_directory[len("gs://") :].split("/")[0]
            path = f"https://storage.googleapis.com/{bucket_name}/{di.path}"
        elif directory.startswith("http"):
            path = di.path
        else:  # local path
            path = "/files" + str((Path(directory) / di.path))
        dataset_item_paths.append(path)

    # Perform clustering
    clustering_model_output = ModelOutput.objects.get(
        pk=clustering_model_output_id, embeddings_path__isnull=False
    )
    params = {
        "embedding_set_path": clustering_model_output.embeddings_path,
        "image_list_path": clustering_model_output.image_list_path,
        "identifiers": internal_identifiers,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/perform_clustering",
        json=params,
    )
    clustering_data = r.json()

    return JsonResponse(
        {
            "paths": dataset_item_paths,
            "identifiers": dataset_item_identifiers,
            "distances": distances,
            "clustering": clustering_data["clustering"],
        }
    )


@api_view(["POST"])
@csrf_exempt
def keep_alive(request):
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/keep_alive",
    )
    return JsonResponse({"status": "success"})


@api_view(["POST"])
@csrf_exempt
def generate_embedding(request):
    payload = json.loads(request.body)
    image_id = payload.get("image_id")
    if image_id:
        payload["identifier"] = DatasetItem.objects.get(pk=image_id).identifier

    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/generate_embedding",
        json=payload,
    )
    return JsonResponse(r.json())


@api_view(["POST"])
@csrf_exempt
def generate_text_embedding(request):
    payload = json.loads(request.body)
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/generate_text_embedding",
        json=payload,
    )
    return JsonResponse(r.json())


@api_view(["POST"])
@csrf_exempt
def query_knn(request, dataset_name):
    payload = json.loads(request.body)
    model_output_id = payload["model_output_id"]
    embeddings = payload["embeddings"]
    use_full_image = bool(payload.get("use_full_image", True))
    use_dot_product = bool(payload.get("use_dot_product", False))

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_output = ModelOutput.objects.get(
        pk=model_output_id, embeddings_path__isnull=False
    )

    query_knn_start = time.time()
    params = {
        "embeddings_path": model_output.embeddings_path,
        "image_list_path": model_output.image_list_path,
        "embeddings": embeddings,
        "use_full_image": use_full_image,
        "use_dot_product": use_dot_product,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_knn",
        json=params,
    )
    response_data = r.json()
    query_knn_end = time.time()
    logger.debug("query_knn time: {:f}".format(query_knn_end - query_knn_start))

    results = process_image_query_results(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set(results, "knn"))


@api_view(["GET"])
@csrf_exempt
def train_svm(request, dataset_name):
    model_output_id = request.GET["model_output_id"]
    pos_tags = parse_tag_set_from_query(request.GET["pos_tags"])
    neg_tags = parse_tag_set_from_query(request.GET.get("neg_tags"))
    augment_negs = bool(
        distutils.util.strtobool(request.GET.get("augment_negs", "false"))
    )

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_output = ModelOutput.objects.get(
        pk=model_output_id, embeddings_path__isnull=False
    )

    pos_dataset_items = DatasetItem.objects.filter(
        tag_sets_to_query(pos_tags),
        dataset=dataset,
        is_val=False,
    )
    neg_dataset_items = DatasetItem.objects.filter(
        tag_sets_to_query(neg_tags),
        dataset=dataset,
        is_val=False,
    ).difference(pos_dataset_items)

    pos_dataset_item_pks = list(pos_dataset_items.values_list("pk", flat=True))
    neg_dataset_item_pks = list(neg_dataset_items.values_list("pk", flat=True))

    # Augment with randomly sampled negatives if requested
    num_extra_negs = settings.SVM_NUM_NEGS_MULTIPLIER * len(pos_dataset_item_pks) - len(
        neg_dataset_item_pks
    )
    if augment_negs and num_extra_negs > 0:
        # Uses "include" and "exclude" category sets from GET request
        all_eligible_pks = filtered_images(
            request, dataset, exclude_pks=pos_dataset_item_pks + neg_dataset_item_pks
        )
        sampled_pks = random.sample(
            all_eligible_pks, min(len(all_eligible_pks), num_extra_negs)
        )
        neg_dataset_item_pks.extend(sampled_pks)

    pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )

    params = {
        "embeddings_path": model_output.embeddings_path,
        "image_list_path": model_output.image_list_path,
        "pos_identifiers": pos_dataset_item_internal_identifiers,
        "neg_identifiers": neg_dataset_item_internal_identifiers,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/train_svm",
        json=params,
    )
    return JsonResponse(r.json())  # {"svm_vector": base64-encoded string}


@api_view(["POST"])
@csrf_exempt
def query_svm(request, dataset_name):
    payload = json.loads(request.body)
    model_output_id = request.GET["model_output_id"]
    svm_vector = payload["svm_vector"]
    score_min = float(payload.get("score_min", 0.0))
    score_max = float(payload.get("score_max", 1.0))

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_output = ModelOutput.objects.get(
        pk=model_output_id, embeddings_path__isnull=False
    )

    params = {
        "embeddings_path": model_output.embeddings_path,
        "image_list_path": model_output.image_list_path,
        "svm_vector": svm_vector,
        "score_min": score_min,
        "score_max": score_max,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_svm",
        json=params,
    )
    response_data = r.json()

    # TODO(mihirg, jeremye): Consider some smarter pagination/filtering scheme to avoid
    # running a separate query over the index every single time the user adjusts score
    # thresholds
    results = process_image_query_results(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set(results, "svm"))


@api_view(["POST"])
@csrf_exempt
def query_ranking(request, dataset_name):
    payload = json.loads(request.body)
    model_output_id = request.GET["model_output_id"]
    score_min = float(payload.get("score_min", 0.0))
    score_max = float(payload.get("score_max", 1.0))

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_output = ModelOutput.objects.get(
        pk=model_output_id, scores_path__isnull=False
    )

    params = {
        "scores_path": model_output.scores_path,
        "image_list_path": model_output.image_list_path,
        "score_min": score_min,
        "score_max": score_max,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_ranking",
        json=params,
    )
    response_data = r.json()

    # TODO(mihirg, jeremye): Consider some smarter pagination/filtering scheme to avoid
    # running a separate query over the index every single time the user adjusts score
    # thresholds
    results = process_image_query_results(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set(results, "ranking"))


@api_view(["POST"])
@csrf_exempt
def query_images(request, dataset_name):
    query_start = time.time()

    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    order = payload.get("order", "id")

    filter_start = time.time()
    result_pks = filtered_images(request, dataset)
    filter_end = time.time()

    if order == "random":
        random.shuffle(result_pks)
    elif order == "id":
        result_pks.sort()
    results = {"pks": result_pks, "distances": [-1 for _ in result_pks]}
    resp = JsonResponse(create_result_set(results, "query"))

    query_end = time.time()
    print(
        f"query_images: tot: {query_end-query_start}, "
        f"filter: {filter_end-filter_start}"
    )
    return resp


#
# ACTIVE VALIDATION
# Note(mihirg): These functions are out of date as of 6/25 and may need to be
# fixed/removed. However, their counterparts on the index server are up to date.
#


VAL_NEGATIVE_TYPE = "model_val_negative"


def get_val_examples(dataset, model_id):
    # Get positive and negative categories
    model = get_object_or_404(DNNModel, model_id=model_id)

    pos_tags = parse_tag_set_from_query(model.category_spec["pos_tags"])
    neg_tags = parse_tag_set_from_query(model.category_spec["neg_tags"])
    augment_negs = model.category_spec.get("augment_negs", False)
    augment_negs_include = (
        parse_tag_set_from_query(model.category_spec.get("augment_negs_include", []))
        if augment_negs
        else set()
    )

    # Limit to validation set
    eligible_dataset_items = DatasetItem.objects.filter(
        dataset=dataset,
        is_val=True,
    )

    # Get positives and negatives matching these categories
    categories = Category.objects.filter(
        tag_sets_to_query(pos_tags, neg_tags, augment_negs_include)
    )
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_dataset_items,
        category__in=categories,
    )
    tags_by_pk = get_tags_from_annotations(annotations)

    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags or t in augment_negs_include for t in tags):
            neg_dataset_item_pks.append(pk)

    # Get extra negatives
    if augment_negs:
        annotations = Annotation.objects.filter(
            dataset_item__in=eligible_dataset_items,
            label_category=model_id,
            label_type=VAL_NEGATIVE_TYPE,
        )
        neg_dataset_item_pks.extend(ann.dataset_item.pk for ann in annotations)

    return pos_dataset_item_pks, neg_dataset_item_pks


@api_view(["POST"])
def query_metrics(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    model_id = payload["model"]
    index_id = payload["index_id"]
    internal_identifiers_to_weights = payload["weights"]  # type: Dict[str, int]

    pos_dataset_item_pks, neg_dataset_item_pks = get_val_examples(dataset, model_id)

    # Construct identifiers, labels, and weights
    dataset_items_by_pk = DatasetItem.objects.in_bulk(
        pos_dataset_item_pks + neg_dataset_item_pks
    )
    identifiers = []
    labels = []
    weights = []
    for pk, label in itertools.chain(
        ((pk, True) for pk in pos_dataset_item_pks),
        ((pk, False) for pk in neg_dataset_item_pks),
    ):
        di = dataset_items_by_pk[pk]
        identifier = di.identifier
        weight = internal_identifiers_to_weights.get(identifier)
        if weight is None:
            continue

        identifiers.append(identifier)
        labels.append(label)
        weights.append(weight)

    # TODO(mihirg): Parse false positives and false negatives
    params = {
        "index_id": index_id,
        "model": model_id,
        "identifiers": identifiers,
        "labels": labels,
        "weights": weights,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_metrics",
        json=params,
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(["POST"])
def query_active_validation(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    model_id = payload["model"]
    index_id = payload["index_id"]
    current_f1 = payload.get("current_f1")
    if current_f1 is None:
        current_f1 = 0.5

    pos_dataset_item_pks, neg_dataset_item_pks = get_val_examples(dataset, model_id)

    # Construct paths, identifiers, and labels
    dataset_items_by_pk = DatasetItem.objects.in_bulk(
        pos_dataset_item_pks + neg_dataset_item_pks
    )
    identifiers = []
    labels = []
    for pk, label in itertools.chain(
        ((pk, True) for pk in pos_dataset_item_pks),
        ((pk, False) for pk in neg_dataset_item_pks),
    ):
        di = dataset_items_by_pk[pk]
        identifiers.append(di.identifier)
        labels.append(label)

    params = {
        "index_id": index_id,
        "model": model_id,
        "identifiers": identifiers,
        "labels": labels,
        "current_f1": current_f1,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_active_validation",
        json=params,
    )
    response_data = r.json()

    if response_data["identifiers"]:
        pks_and_paths = list(
            DatasetItem.objects.filter(
                dataset=dataset,
                identifier__in=response_data["identifiers"],
                is_val=True,
            ).values_list("pk", "path")
        )
        random.shuffle(pks_and_paths)
        pks, paths = zip(*pks_and_paths)
    else:
        pks, paths = [], []

    bucket_name = dataset.val_directory[len("gs://") :].split("/")[0]
    path_template = "https://storage.googleapis.com/{:s}/".format(bucket_name) + "{:s}"
    paths = [path_template.format(p) for p in paths]

    return JsonResponse(
        {
            "paths": paths,
            "identifiers": pks,
            "weights": response_data["weights"],
        }
    )


@api_view(["POST"])
def add_val_annotations(request):
    payload = json.loads(request.body)
    annotations = payload["annotations"]
    user_email = payload["user"]
    model = payload["model"]

    anns = []
    cat_modes = defaultdict(int)
    dataset = None
    for ann_payload in annotations:
        image_pk = ann_payload["identifier"]
        is_other_negative = ann_payload.get("is_other_negative", False)
        mode_str = "NEGATIVE" if is_other_negative else ann_payload["mode"]
        category_name = (
            "active:" + model if is_other_negative else ann_payload["category"]
        )

        user, _ = User.objects.get_or_create(email=user_email)
        category, _ = Category.objects.get_or_create(name=category_name)
        mode, _ = Mode.objects.get_or_create(name=mode_str)

        di = DatasetItem.objects.get(pk=image_pk)
        dataset = di.dataset
        assert di.is_val
        ann = Annotation(
            dataset_item=di,
            user=user,
            category=category,
            mode=mode,
            misc_data={"created_by": "active_val"},
        )
        cat_modes[(category, mode)] += 1
        anns.append(ann)

    Annotation.objects.bulk_create(anns)
    for (cat, mode), c in cat_modes.items():
        category_count, _ = CategoryCount.objects.get_or_create(
            dataset=dataset, category=cat, mode=mode
        )
        category_count.count += c
        category_count.save()
    return JsonResponse({"created": len(anns)})


# DATASET INFO


@api_view(["GET"])
@csrf_exempt
def get_datasets(request):
    datasets = Dataset.objects.filter(hidden=False)
    dataset_names = list(datasets.values_list("name", flat=True))
    return JsonResponse({"dataset_names": dataset_names})


@api_view(["GET"])
@csrf_exempt
def get_dataset_info(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    num_train = dataset.datasetitem_set.filter(is_val=False).count()
    num_val = dataset.datasetitem_set.filter(is_val=True).count()

    return JsonResponse(
        {
            "num_train": num_train,
            "num_val": num_val,
        }
    )


@api_view(["POST"])
def add_model_output(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    payload = json.loads(request.body)
    name = payload["name"]

    embeddings_path = payload.get("embeddings_path", "")
    scores_path = payload.get("scores_path", "")

    if embeddings_path == "" and scores_path == "":
        return JsonResponse(
            {
                "status": "failed",
                "reason": "Must supply one of 'embeddings_path' or 'scores_path'",
            }
        )

    if "paths" in payload:
        path_to_di = {
            di.path: di
            for di in DatasetItem.objects.filter(
                dataset=dataset, path__in=payload["paths"]
            )
        }

        FORAGER_IMAGE_LISTS_DIR.mkdir(parents=True, exist_ok=True)

        model_uuid = str(uuid.uuid4())

        image_list_path = FORAGER_IMAGE_LISTS_DIR / model_uuid
        with open(image_list_path, "w") as f:
            for path in payload["paths"]:
                di = path_to_di[path]
                split = "val" if di.is_val else "train"
                f.write(f"{split} {di.identifier} {path}\n")

    elif "image_list_path" in payload:
        image_list_path = payload["image_list_path"]
    else:
        return JsonResponse(
            {
                "status": "failed",
                "reason": "Must supply one of 'paths', or 'image_list_path'.",
            },
            status=400,
        )

    model_output = ModelOutput(
        dataset=dataset,
        name=name,
        embeddings_path=embeddings_path,
        scores_path=scores_path,
        image_list_path=image_list_path,
    )
    model_output.save()
    return JsonResponse({"status": "success"})


@api_view(["GET"])
def get_model_outputs(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_output_objs = ModelOutput.objects.filter(dataset=dataset)
    return JsonResponse(
        {"model_outputs": list(map(model_output_info, model_output_objs))}
    )


def model_output_info(model_output):
    return {
        "id": model_output.pk,
        "name": model_output.name,
        "has_embeddings": model_output.embeddings_path != "",
        "has_scores": model_output.scores_path != "",
        "timestamp": model_output.last_updated,
    }


@api_view(["DELETE"])
def delete_model_output(request, model_output_id):
    model_output = ModelOutput.object.get(pk=model_output_id)
    model_output.delete()
    return JsonResponse({"status": "success"})


@api_view(["GET"])
@csrf_exempt
def get_models(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    model_objs = DNNModel.objects.filter(
        dataset=dataset,
        checkpoint_path__isnull=False,
    ).order_by("-last_updated")

    model_names = set()
    latest = {}
    with_output = {}
    for model in model_objs:
        model_names.add(model.name)
        if model.name not in latest:
            latest[model.name] = model
        if model.output_directory and model.name not in with_output:
            with_output[model.name] = model

    models = [
        {
            "name": model_name,
            "latest": model_info(latest[model_name]),
            "with_output": model_info(with_output.get(model_name)),
        }
        for model_name in model_names
    ]
    return JsonResponse({"models": models})


def model_info(model):
    if model is None:
        return None

    pos_tags = parse_tag_set_from_query(model.category_spec.get("pos_tags", []))
    neg_tags = parse_tag_set_from_query(model.category_spec.get("neg_tags", []))
    augment_negs_include = parse_tag_set_from_query(
        model.category_spec.get("augment_negs_include", [])
    )
    return {
        "model_id": model.model_id,
        "timestamp": model.last_updated,
        "has_checkpoint": model.checkpoint_path is not None,
        "has_output": model.output_directory is not None,
        "pos_tags": serialize_tag_set_for_client(pos_tags),
        "neg_tags": serialize_tag_set_for_client(neg_tags | augment_negs_include),
        "augment_negs": model.category_spec.get("augment_negs", False),
        "epoch": model.epoch,
    }


@api_view(["POST"])
@csrf_exempt
def create_dataset(request):
    payload = json.loads(request.body)
    name = payload["dataset"]
    train_directory = payload["train_images_directory"]
    val_directory = payload["val_images_directory"]

    # Get all paths for train and val
    paths: List[Tuple[str, str, bool]] = []
    for directory, is_val in [(train_directory, False), (val_directory, True)]:
        for root, dirs, files in os.walk(os.path.expanduser(directory)):
            image_paths = [
                (os.path.abspath(os.path.join(root, path)), str(uuid.uuid4()), is_val)
                for path in files
                if (
                    path.endswith(".jpg")
                    or path.endswith(".JPG")
                    or path.endswith(".jpeg")
                    or path.endswith(".JPEG")
                    or path.endswith(".png")
                    or path.endswith(".PNG")
                )
            ]
            paths += image_paths

    # Add dataset to db
    dataset = Dataset(
        name=name,
        train_directory=train_directory,
        val_directory=val_directory,
    )
    dataset.save()

    # Create all the DatasetItems for this dataset
    items = [
        DatasetItem(
            dataset=dataset,
            identifier=ident,
            path=path,
            is_val=is_val,
        )
        for path, ident, is_val in paths
    ]
    DatasetItem.objects.bulk_create(items, batch_size=10000)

    # Compute resnet and clip model outputs if requested
    splits_to_image_paths = {"train": [], "val": []}
    for path, ident, is_val in paths:
        splits_to_image_paths["val" if is_val else "train"].append((path, ident))

    try:
        r = requests.post(
            settings.EMBEDDING_SERVER_ADDRESS + "/start_embedding_job",
            json={
                "model_output_name": "resnet",
                "splits_to_image_paths": splits_to_image_paths,
                "embedding_type": "resnet",
            },
        )
        response_data = r.json()
        resnet_job_id = response_data["job_id"]

        r = requests.post(
            settings.EMBEDDING_SERVER_ADDRESS + "/start_embedding_job",
            json={
                "model_output_name": "clip",
                "splits_to_image_paths": splits_to_image_paths,
                "embedding_type": "clip",
            },
        )
        response_data = r.json()
        clip_job_id = response_data["job_id"]

        job_ids_left = [("resnet", resnet_job_id), ("clip", clip_job_id)]
        while len(job_ids_left) > 0:
            next_jobs = []
            for name, job_id in job_ids_left:
                r = requests.get(
                    settings.EMBEDDING_SERVER_ADDRESS + "/embedding_job_status",
                    params={"job_id": job_id},
                )
                response_data = r.json()
                if not response_data["has_job"]:
                    dataset.delete()
                    return JsonResponse(
                        {
                            "status": "failure",
                            "failure_reason": "Embedding server did not receive embedding job",
                        },
                        status=500,
                    )

                if not response_data["finished"]:
                    next_jobs.append((name, job_id))
                else:
                    if response_data["failed"]:
                        failure_reason = response_data["failure_reason"]
                        dataset.delete()
                        return JsonResponse(
                            {"status": "failure", "failure_reason": failure_reason},
                            status=500,
                        )
                    ModelOutput(
                        dataset=dataset,
                        name=name,
                        image_list_path=response_data["image_list_path"],
                        embeddings_path=response_data["embeddings_path"],
                    ).save()
            time.sleep(3)
            job_ids_left = next_jobs

    except requests.exceptions.RequestException as e:  # This is the correct syntax
        dataset.delete()
        return JsonResponse(
            {"status": "failure", "reason": "Failed to create embeddings for dataset."},
            status=500,
        )

    # Add model outputs to db

    return JsonResponse({"status": "success"})


@api_view(["POST"])
@csrf_exempt
def delete_dataset(request):
    payload = json.loads(request.body)
    dataset_name = payload["dataset"]

    dataset = get_object_or_404(Dataset, name=dataset_name)
    dataset.delete()

    return JsonResponse({"status": "success"})


@api_view(["POST"])
@csrf_exempt
def get_annotations(request):
    payload = json.loads(request.body)

    annotations = Annotation.objects.all()
    if "dataset_name" in payload:
        dataset = get_object_or_404(Dataset, name=payload["dataset_name"])
        annotations = annotations.filter(
            dataset_item__in=DatasetItem.objects.filter(dataset=dataset)
        )

    if "identifiers" in payload:
        image_pks = [i for i in payload["identifiers"] if i]
        annotations = annotations.filter(
            dataset_item__in=DatasetItem.objects.filter(pk__in=image_pks),
        )

    if "tags" in payload:
        annotations = annotations.filter(
            category__in=Category.objects.filter(name__in=payload["tags"])
        )

    if "modes" in payload:
        annotations = annotations.filter(
            mode__in=Mode.objects.filter(name__in=payload["modes"])
        )

    tags_by_pk = get_tags_from_annotations(annotations)
    boxes_by_pk = get_boxes_from_annotations(annotations)
    annotations_by_pk = defaultdict(lambda: {"tags": [], "boxes": []})
    for pk, tags in tags_by_pk.items():
        annotations_by_pk[pk]["tags"] = serialize_tag_set_for_client(tags)
    for pk, boxes in boxes_by_pk.items():
        annotations_by_pk[pk]["boxes"] = serialize_boxes_for_client(boxes)
    if "by_path" in payload and payload["by_path"]:
        pk_to_path = {
            a["pk"]: a["path"]
            for a in DatasetItem.objects.filter(
                pk__in=list(annotations_by_pk.keys())
            ).values("pk", "path")
        }
        annotations_by_path = {
            pk_to_path[pk]: [
                {"category": tag["category"], "mode": tag["value"]} for tag in v["tags"]
            ]
            for pk, v in annotations_by_pk.items()
        }
        return JsonResponse(annotations_by_path)
    else:
        return JsonResponse(annotations_by_pk)


@api_view(["POST"])
@csrf_exempt
def add_annotations(request):
    payload = json.loads(request.body)
    image_pks = payload["identifiers"]
    images = DatasetItem.objects.filter(pk__in=image_pks)
    num_created = bulk_add_single_tag_annotations(payload, images)
    return JsonResponse({"created": num_created})


@api_view(["POST"])
@csrf_exempt
def add_annotations_multi(request):
    payload = json.loads(request.body)
    num_created = bulk_add_multi_annotations(payload)
    return JsonResponse({"created": num_created})


@api_view(["POST"])
@csrf_exempt
def add_annotations_by_internal_identifiers(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    payload = json.loads(request.body)
    image_identifiers = payload["identifiers"]
    images = DatasetItem.objects.filter(
        dataset=dataset, identifier__in=image_identifiers
    )
    num_created = bulk_add_single_tag_annotations(payload, images)
    return JsonResponse({"created": num_created})


@api_view(["POST"])
@csrf_exempt
def add_annotations_to_result_set(request):
    payload = json.loads(request.body)
    result_set_id = payload["result_set_id"]
    lower_bound = float(payload["from"])
    upper_bound = float(payload["to"])

    result_set = current_result_sets[result_set_id]
    result_ranking = result_set.ranking
    # e.g., lower_bound=0.0, upper_bound=0.5 -> second half of the result set
    start_index = math.ceil(len(result_ranking) * (1.0 - upper_bound))
    end_index = math.floor(len(result_ranking) * (1.0 - lower_bound))
    image_pks = result_ranking[start_index:end_index]

    images = DatasetItem.objects.filter(pk__in=image_pks)
    num_created = bulk_add_single_tag_annotations(payload, images)
    return JsonResponse({"created": num_created})


def bulk_add_single_tag_annotations(payload, images):
    """Adds annotations for a single tag to many dataset items"""
    if not images:
        return 0

    user_email = payload["user"]
    category_name = payload["category"]
    mode_name = payload["mode"]
    created_by = payload.get("created_by", "tag" if len(images) == 1 else "tag-bulk")

    dataset = images[0].dataset

    user, _ = User.objects.get_or_create(email=user_email)
    category, _ = Category.objects.get_or_create(name=category_name)

    Annotation.objects.filter(
        dataset_item__in=images, category=category, is_box=False
    ).delete()

    # HACK(mihirg): We don't have an actual endpoint to delete annotations, so deletion
    # is currently signalled by the frontend sending an "add annotation" request with
    # the mode "TOMBSTOME"
    if mode_name == "TOMBSTONE":
        return 0

    mode, _ = Mode.objects.get_or_create(name=mode_name)
    annotations = [
        Annotation(
            dataset_item=di,
            user=user,
            category=category,
            mode=mode,
            is_box=False,
            misc_data={"created_by": created_by},
        )
        for di in images
    ]
    bulk_add_annotations(dataset, annotations)

    return len(annotations)


def bulk_add_multi_annotations(payload: Dict):
    """Adds multiple annotations for the same dataset and user to the database
    at once"""
    dataset_name = payload["dataset"]
    dataset = get_object_or_404(Dataset, name=dataset_name)
    user_email = payload["user"]
    user, _ = User.objects.get_or_create(email=user_email)
    created_by = payload.get(
        "created_by", "tag" if len(payload["annotations"]) == 1 else "tag-bulk"
    )

    # Get pks
    idents = [
        ann["identifier"] for ann in payload["annotations"] if "identifier" in ann
    ]
    di_pks = list(
        DatasetItem.objects.filter(dataset=dataset, identifier__in=idents).values_list(
            "pk", "identifier"
        )
    )
    ident_to_pk = {ident: pk for pk, ident in di_pks}

    paths = [ann["path"] for ann in payload["annotations"] if "path" in ann]
    di_pks = list(
        DatasetItem.objects.filter(dataset=dataset, path__in=paths).values_list(
            "pk", "path"
        )
    )
    path_to_pk = {path: pk for pk, path in di_pks}

    cats = {}
    modes = {}
    to_delete = defaultdict(set)
    annotations = []
    for ann in payload["annotations"]:
        db_ann = Annotation()
        category_name = ann["category"]
        mode_name = ann["mode"]
        is_box = ann.get("is_box", False)
        if "identifier" in ann:
            pk = ident_to_pk[ann["identifier"]]
        elif "pk" in ann:
            pk = ann["pk"]
        elif "path" in ann:
            pk = path_to_pk[ann["path"]]
        else:
            raise ValueError(
                "All annotations must have one of 'identifier', 'pk', or 'path'."
            )

        if category_name not in cats:
            cats[category_name] = Category.objects.get_or_create(name=category_name)[0]
        if not is_box:
            to_delete[cats[category_name]].add(pk)

        # HACK(mihirg): We don't have an actual endpoint to delete annotations, so
        # deletion is currently signalled by the frontend sending an "add annotation"
        # request with the mode "TOMBSTOME"
        if mode_name == "TOMBSTONE":
            continue

        if mode_name not in modes:
            modes[mode_name] = Mode.objects.get_or_create(name=mode_name)[0]

        db_ann.category = cats[category_name]
        db_ann.mode = modes[mode_name]
        db_ann.dataset_item_id = pk
        db_ann.user = user
        db_ann.is_box = is_box

        if db_ann.is_box:
            db_ann.bbox_x1 = ann["x1"]
            db_ann.bbox_y1 = ann["y1"]
            db_ann.bbox_x2 = ann["x2"]
            db_ann.bbox_y2 = ann["y2"]

        db_ann.misc_data = {"created_by": created_by}
        annotations.append(db_ann)

    for cat, pks in to_delete.items():
        # Delete per-frame annotations for the category if they exist since
        # we should only have on mode per image
        Annotation.objects.filter(
            category=cat, dataset_item_id__in=pks, is_box=False
        ).delete()

    bulk_add_annotations(dataset, annotations)

    return len(annotations)


def bulk_add_annotations(dataset, annotations):
    """Handles book keeping for adding many annotations at once"""
    Annotation.objects.bulk_create(annotations)
    counts = defaultdict(int)
    for ann in annotations:
        counts[(ann.category, ann.mode)] += 1

    for (cat, mode), count in counts.items():
        category_count, _ = CategoryCount.objects.get_or_create(
            dataset=dataset, category=cat, mode=mode
        )
        category_count.count += count
        category_count.save()


@api_view(["POST"])
@csrf_exempt
def delete_category(request):
    payload = json.loads(request.body)
    category = payload["category"]

    category = Category.objects.get(name=category)
    category.delete()

    return JsonResponse({"status": "success"})


@api_view(["POST"])
@csrf_exempt
def update_category(request):
    payload = json.loads(request.body)

    old_category_name = payload["oldCategory"]
    new_category_name = payload["newCategory"]

    category = Category.objects.get(name=old_category_name)
    category.name = new_category_name
    category.save()

    return JsonResponse({"status": "success"})


@api_view(["GET"])
@csrf_exempt
def get_category_counts(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    counts = CategoryCount.objects.filter(dataset=dataset).values(
        "category__name", "mode__name", "count"
    )
    n_labeled = defaultdict(dict)
    for c in counts:
        category = c["category__name"]
        mode = c["mode__name"]
        count = c["count"]
        n_labeled[category][mode] = count

    return JsonResponse(n_labeled)
