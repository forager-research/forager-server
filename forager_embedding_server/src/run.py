import asyncio
import base64
import concurrent
from collections import defaultdict
from dataclasses import dataclass
import functools
import heapq
import itertools
from io import BytesIO
import json
import logging
import math
import operator
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import aiohttp
from dataclasses_json import dataclass_json
import fastcluster
from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image
from sanic import Sanic
import sanic.response as resp
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

from typing import Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Iterable

from interactive_index import InteractiveIndex
from interactive_index.utils import sample_farthest_vectors

from knn import utils
from knn.clusters import TerraformModule
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer, ListReducer, IsFinishedReducer, VectorReducer
from knn.utils import JSONType

import ais
import config
from index_jobs import (
    AdderReducer,
    IndexType,
    MapperReducer,
    Trainer,
    TrainingJob,
    LocalFlatIndex,
)
from bgsplit_jobs import (
    BGSplitTrainingJob,
    BGSplitInferenceJob,
)
from utils import CleanupDict


# Create a logger for the server
logger = logging.getLogger("index_server")
logger.setLevel(logging.DEBUG)

# Create a file handler for the log
log_fh = logging.FileHandler("index_server.log")
log_fh.setLevel(logging.DEBUG)

# Create a console handler to print errors to console
log_ch = logging.StreamHandler()
log_ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_fh.setFormatter(formatter)
log_ch.setFormatter(formatter)

# Attach handlers
logger.addHandler(log_fh)
logger.addHandler(log_ch)


class LabeledIndex:
    LABELS_FILENAME = "labels.json"
    TRAIN_IDENTIFIERS_FILENAME = "identifiers.json"
    VAL_IDENTIFIERS_FILENAME = "val_identifiers.json"

    @dataclass_json
    @dataclass
    class QueryResult:
        id: int
        dist: float = 0.0
        spatial_dists: Optional[List[Tuple[int, float]]] = None
        label: str = ""

    @dataclass
    class FurthestQueryResult:
        id: int
        spatial_locs: Optional[List[int]] = None
        label: str = ""

    # Don't use this directly - use a @classmethod constructor
    def __init__(self, index_id: str, *args, **kwargs):
        self.index_id = index_id
        self.index_dir = config.INDEX_PARENT_DIR / self.index_id
        self.logger = logging.getLogger(
            f"index_server.LabeledIndex({self.index_id[:6]})"
        )
        self.ready = asyncio.Event()

        # Will be filled by each individual constructor
        # TODO(mihirg): Use primary key instead of identifiers or labels
        self.labels: List[str] = []
        self.train_identifiers: Optional[Dict[str, int]] = None
        self.val_identifiers: Optional[Dict[str, int]] = None
        self.indexes: Dict[IndexType, InteractiveIndex] = {}
        self.local_flat_indexes: Dict[str, LocalFlatIndex] = {}

        # Will only be used by the start_building() pathway
        self.bucket: Optional[str] = None
        self.cluster: Optional[TerraformModule] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.mapper_job: Optional[MapReduceJob] = None
        self.build_local_flat_index_task: Optional[asyncio.Task] = None
        self.training_jobs: CleanupDict[IndexType, TrainingJob] = CleanupDict(
            lambda job: job.stop()
        )
        self.start_adding_eventually_task: Optional[asyncio.Task] = None
        self.adder_job: Optional[MapReduceJob] = None
        self.resizer_job: Optional[MapReduceJob] = None
        self.merge_task: Optional[asyncio.Task] = None
        self.cluster_unlock_fn: Optional[Callable[[None], None]] = None

    def get_local_flat_index(self, model: str):
        if model not in self.local_flat_indexes:
            # TODO(mihirg, fpoms): Don't hardcode path and dims here?
            self.local_flat_indexes[model] = LocalFlatIndex.load(
                config.MODEL_OUTPUTS_PARENT_DIR / model,
                len(self.labels),
                config.BGSPLIT_EMBEDDING_DIM,
            )
        index = self.local_flat_indexes[model]
        return index

    # NOTE(mihirg): We don't normalize scores here because this currently only runs on
    # model outputs that are 0 to 1 anyway; consider adding as a param in the future
    def rank_brute_force(
        self, model: str, min_s: float = 0.0, max_s: float = 1.0
    ) -> List[QueryResult]:
        local_flat_index = self.get_local_flat_index(model)
        assert local_flat_index.scores is not None

        ranking = np.argsort(local_flat_index.scores)[::-1]
        sorted_results = []
        for i in ranking:
            i = int(i)
            s = float(local_flat_index.scores[i])
            if min_s <= s <= max_s:
                sorted_results.append(LabeledIndex.QueryResult(i, s))

        for result in sorted_results:
            result.label = self.labels[result.id]
        return sorted_results

    def query_brute_force(
        self,
        query_vector: np.ndarray,
        dot_product: bool = False,
        min_d: float = 0.0,
        max_d: float = 1.0,
        chunk_size: int = config.BRUTE_FORCE_QUERY_CHUNK_SIZE,
        model: str = config.DEFAULT_QUERY_MODEL,
    ) -> List[QueryResult]:
        local_flat_index = self.get_local_flat_index(model)

        self.logger.info(f"Brute force query: dot_product={dot_product}")
        start = time.perf_counter()

        # TODO(mihirg): Process CHUNK_SIZE rows at a time for large datasets
        if dot_product:
            dists = local_flat_index.index @ query_vector
        else:
            dists = cdist(np.expand_dims(query_vector, axis=0), local_flat_index.index)
            dists = np.squeeze(dists, axis=0)

        sorted_results = []
        lowest_dist = np.min(dists)
        highest_dist = np.max(dists)

        for i, d in enumerate(dists):
            d = float(d)
            d = (d - lowest_dist) / (highest_dist - lowest_dist)  # normalize
            if min_d <= d <= max_d:
                sorted_results.append(LabeledIndex.QueryResult(i, d))

        sorted_results.sort(key=operator.attrgetter("dist"), reverse=dot_product)

        end = time.perf_counter()
        self.logger.debug(
            f"Brute force query of size {query_vector.shape} with "
            f"n_vectors={len(self.labels)} took {end - start:.3f}s, and "
            f"got {len(sorted_results)} results."
        )

        for result in sorted_results:
            result.label = self.labels[result.id]
        return sorted_results

    def query(
        self,
        query_vector: np.ndarray,
        num_results: Optional[int] = None,  # if None, all results
        num_probes: Optional[int] = None,
        use_full_image: bool = False,
        svm: bool = False,
        min_d: float = 0.0,
        max_d: float = 1.0,
    ) -> List[QueryResult]:
        assert self.ready.is_set()

        self.logger.info(f"Query: use_full_image={use_full_image}, svm={svm}")
        start = time.perf_counter()

        if use_full_image:
            index = self.indexes[IndexType.FULL_DOT if svm else IndexType.FULL]
            if num_results is None:
                num_results = len(self.labels)  # can't use n_vectors - distributed add
                num_probes = index.n_centroids

            dists, (ids, _) = index.query(
                query_vector, num_results, n_probes=num_probes
            )
            assert len(ids) == 1 and len(dists) == 1
            lowest_dist = np.min(dists)
            highest_dist = np.max(dists)

            sorted_results = []
            for i, d in zip(ids[0], dists[0]):
                i, d = int(i), float(d)  # cast numpy types
                d = (d - lowest_dist) / (highest_dist - lowest_dist)  # normalize
                if svm:
                    d = 1.0 - d  # invert
                if i >= 0 and min_d <= d <= max_d:
                    sorted_results.append(LabeledIndex.QueryResult(i, d))
        else:
            assert (
                min_d == 0.0 and max_d == 1.0
            ), "Distance bounds not supported for spatial queries"

            index = self.indexes[IndexType.SPATIAL_DOT if svm else IndexType.SPATIAL]
            if num_results is None:
                # TODO(mihirg): Set num_results properly
                num_results = config.QUERY_NUM_RESULTS_MULTIPLE * len(self.labels)
                num_probes = index.n_centroids

            dists, (ids, locs) = index.query(
                query_vector,
                config.QUERY_NUM_RESULTS_MULTIPLE * num_results,
                n_probes=num_probes,
            )
            assert len(ids) == 1 and len(locs) == 1 and len(dists) == 1

            # Gather lowest QUERY_PATCHES_PER_IMAGE distances for each image
            dists_by_id: DefaultDict[int, List[float]] = defaultdict(list)
            spatial_dists_by_id: DefaultDict[
                int, List[Tuple[int, float]]
            ] = defaultdict(list)
            for i, l, d in zip(ids[0], locs[0], dists[0]):
                i, l, d = int(i), int(l), float(d)  # cast numpy types
                if i >= 0 and len(dists_by_id[i]) < config.QUERY_PATCHES_PER_IMAGE:
                    dists_by_id[i].append(d)
                    spatial_dists_by_id[i].append((l, d))

            # Average them and resort
            result_gen = (
                LabeledIndex.QueryResult(i, sum(ds) / len(ds), spatial_dists_by_id[i])
                for i, ds in dists_by_id.items()
                if len(ds) == config.QUERY_PATCHES_PER_IMAGE
            )
            sorted_results = heapq.nsmallest(
                num_results, result_gen, operator.attrgetter("dist")
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Query of size {query_vector.shape} with k={num_results}, "
            f"n_probes={num_probes}, n_centroids={index.n_centroids}, and "
            f"n_vectors={index.n_vectors} took {end - start:.3f}s, and "
            f"got {len(sorted_results)} results."
        )

        for result in sorted_results:
            result.label = self.labels[result.id]
        return sorted_results

    def query_farthest(
        self,
        query_vector: np.ndarray,
        fraction: float,
        max_samples: int,
        use_full_image: bool = False,
    ) -> List[FurthestQueryResult]:
        if use_full_image:
            ids, _ = sample_farthest_vectors(
                self.indexes[IndexType.FULL_DOT], query_vector, fraction, max_samples
            )
            results = [LabeledIndex.FurthestQueryResult(int(i)) for i in ids]
        else:
            ids, locs = sample_farthest_vectors(
                self.indexes[IndexType.SPATIAL_DOT],
                query_vector,
                fraction,
                config.QUERY_NUM_RESULTS_MULTIPLE * max_samples,
            )

            # Gather spatial locations for each image
            locs_by_id: DefaultDict[int, List[int]] = defaultdict(list)
            for i, l in zip(ids, locs):
                i, l = int(i), int(l)  # cast numpy type  # noqa: E741
                locs_by_id[i].append(l)

            # Return up to max_samples images with the highest number (but at least
            # QUERY_PATCHES_PER_IMAGE) of returned spatial locations
            result_gen = (
                LabeledIndex.FurthestQueryResult(i, locs)
                for i, locs in locs_by_id.items()
            )
            results = heapq.nlargest(
                max_samples, result_gen, lambda r: len(r.spatial_locs)
            )

        for result in results:
            result.label = self.labels[result.id]
        return results

    def get_train_identifiers(self) -> List[str]:
        assert self.train_identifiers is not None
        return list(self.train_identifiers.keys())

    def get_val_identifiers(self) -> List[str]:
        assert self.val_identifiers is not None
        return list(self.val_identifiers.keys())

    def identifiers_to_inds(self, identifiers: Iterable[str]) -> List[int]:
        assert self.train_identifiers is not None
        assert self.val_identifiers is not None
        inds = [
            self.train_identifiers[id]
            if id in self.train_identifiers
            else self.val_identifiers[id]
            for id in identifiers
        ]
        return inds

    def get_embeddings(
        self, identifiers: Iterable[str], model: str = config.DEFAULT_QUERY_MODEL
    ) -> np.ndarray:
        start_time = time.perf_counter()
        local_flat_index = self.get_local_flat_index(model)
        middle_time = time.perf_counter()
        inds = self.identifiers_to_inds(identifiers)
        end_time = time.perf_counter()
        print(
            f"get_local_flat_index ({model}) took {middle_time - start_time}; identifiers_to_inds took {end_time - middle_time}"
        )
        return local_flat_index.index[inds]

    def cluster_identifiers(
        self, identifiers: Iterable[str], model: str = config.DEFAULT_QUERY_MODEL
    ) -> List[List[float]]:
        start_time = time.perf_counter()
        embeddings = self.get_embeddings(identifiers, model)
        middle_time = time.perf_counter()
        ret = self._cluster(embeddings)
        end_time = time.perf_counter()
        print(
            f"Clustering took {end_time - start_time} seconds - {middle_time - start_time} to read embeddings + {end_time - middle_time} to perform clustering"
        )
        return ret

    def _cluster(self, embeddings: np.ndarray) -> List[List[float]]:
        # Perform hierarchical clustering
        result = fastcluster.linkage(embeddings, method="ward", preserve_input=False)
        max_dist = result[-1, 2]

        # Simplify dendogram matrix by using original cluster indexes
        simplified = []
        clusters = list(range(len(embeddings)))
        for a, b, dist, _ in result:
            a, b = int(a), int(b)
            simplified.append([clusters[a], clusters[b], dist / max_dist])
            clusters.append(clusters[a])
        return simplified

    def get_model_scores(
        self, model: str, identifiers: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        local_flat_index = self.get_local_flat_index(model)
        assert local_flat_index.scores is not None
        if identifiers is None:
            return local_flat_index.scores

        inds = self.identifiers_to_inds(identifiers)
        return local_flat_index.scores[inds]

    # CLEANUP

    def delete(self):
        assert self.ready.is_set()
        shutil.rmtree(self.index_dir)

    async def stop_building(self):
        # Map
        if self.mapper_job:
            await self.mapper_job.stop()
        if (
            self.build_local_flat_index_task
            and not self.build_local_flat_index_task.done()
        ):
            self.build_local_flat_index_task.cancel()
            await self.build_local_flat_index_task

        # Train
        await self.training_jobs.clear_async()

        # Add
        if (
            self.start_adding_eventually_task
            and not self.start_adding_eventually_task.done()
        ):
            self.start_adding_eventually_task.cancel()
            await self.start_adding_eventually_task
        if self.adder_job:
            await self.adder_job.stop()

        # Resize
        if self.resizer_job:
            await self.resizer_job.stop()

        # Close network connections
        if self.http_session:
            await self.http_session.close()

        # Unlock cluster
        if self.cluster_unlock_fn:
            self.cluster_unlock_fn()

        # Merge
        if self.merge_task:
            self.merge_task.cancel()
            await self.merge_task

        # Delete unnecessary intermediates from local disk
        if not self.ready.is_set():
            shutil.rmtree(self.index_dir)

    # INDEX CREATION
    # Note(mihirg): This is very broken, since we have not been keeping it up to date
    # with changes to the data model. Don't run this without fixing, and probably
    # talking to me.

    @classmethod
    async def start_building(
        cls,
        cluster: TerraformModule,
        cluster_unlock_fn: Callable[[], None],
        bucket: str,
        paths: List[str],
        identifiers: List[str],
        *args,
        **kwargs,
    ):
        self = cls(str(uuid.uuid4()), *args, **kwargs)
        self.bucket = bucket
        self.http_session = utils.create_unlimited_aiohttp_session()

        # Randomly shuffle input images
        inds = np.arange(len(paths))
        np.random.shuffle(inds)
        self.labels = [paths[i] for i in inds]
        self.train_identifiers = {identifiers[i]: new_i for new_i, i in enumerate(inds)}
        self.val_identifiers = {}  # TODO(mihirg): fix this to include val set
        iterable = (
            {"id": i, "image": path, "augmentations": {}}
            for i, path in enumerate(self.labels)
        )

        # Wait for the cluster to start, then do some configuration for the Train step,
        # which will start automatically as soon as the Map step (below) has made
        # sufficient progress
        await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())
        self.cluster = cluster
        self.cluster_unlock_fn = cluster_unlock_fn

        trainers = [Trainer(url) for url in self.cluster.output["trainer_urls"]]
        for index_type in IndexType:
            self.training_jobs[index_type] = TrainingJob(
                index_type,
                len(paths),
                self.index_id,
                trainers[index_type.value % len(trainers)],
                self.cluster.mount_parent_dir,
                self.http_session,
            )

        # Step 1: "Map" input images to embedding files saved to shared disk
        # TODO(mihirg): Fail gracefully if entire Map, Train, Add, or Resize jobs fail
        notification_request_to_configure_indexes = MapperReducer.NotificationRequest(
            self.configure_indexes,
            on_num_images=config.NUM_IMAGES_TO_MAP_BEFORE_CONFIGURING_INDEX,
        )

        nproc = self.cluster.output["mapper_nproc"]
        n_mappers = int(
            self.cluster.output["num_mappers"] * config.MAPPER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.MAPPER_CHUNK_SIZE(nproc)
        self.mapper_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["mapper_url"],
                n_mappers=n_mappers,
            ),
            MapperReducer([notification_request_to_configure_indexes]),
            {"input_bucket": self.bucket, "return_type": "save"},
            session=self.http_session,
            n_retries=config.MAPPER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.MAPPER_REQUEST_TIMEOUT,
        )
        await self.mapper_job.start(iterable, self.start_training, len(paths))
        self.logger.info(f"Map: started with {len(paths)} images")

        # Start a background task that consumes Map outputs as they're generated and
        # builds a local flat index of full-image embeddings
        self.local_flat_index = LocalFlatIndex.create(
            self.index_dir,
            len(self.labels),
            config.EMBEDDING_DIM,
            self.cluster.mount_parent_dir,
        )
        self.build_local_flat_index_task = self.build_local_flat_index_in_background()

        # Start a background task that waits until the Train step (started automatically
        # per above) is done and then kicks off the Add step
        self.start_adding_eventually_task = self.start_adding_eventually()

        return self

    def configure_indexes(self, mapper_result: MapperReducer.Result):
        self.logger.debug(
            "Map: configuring indexes after successfully processing "
            f"{mapper_result.num_images} images"
        )

        # Once we've successfully computed embeddings for a few images, use the results
        # so far to configure the indexes (number of centroids, etc.), and then use that
        # index configuration to figure out the number of images/embeddings we need to
        # start training each index. We can't do this before starting the Map step
        # because, for the spatial indexes, we need to know how many spatial embeddings
        # there are per image (dependent on model and image resolution) in order to
        # estimate the total number of vectors that will be in the index, which informs
        # the index configuration.
        assert self.mapper_job
        for index_type, job in self.training_jobs.items():
            self.mapper_job.reducer.add_notification_request(
                job.make_notification_request_to_start_training(
                    mapper_result,
                    functools.partial(self.start_training, index_type=index_type),
                )
            )

    @utils.unasync_as_task
    async def build_local_flat_index_in_background(self):
        # Step 2: As the Map step runs, build a local flat index of the full-image
        # embeddings it generates to facilitate fast SVM queries and a distance matrix
        # to facilitate fast clustering
        with concurrent.futures.ThreadPoolExecutor(
            config.LOCAL_INDEX_BUILDING_NUM_THREADS
        ) as pool:
            coro_gen = (
                utils.run_in_executor(
                    self.local_flat_index.add_from_file, path_tmpl, executor=pool
                )
                async for path_tmpl in self.mapper_job.reducer.output_path_tmpl_gen()
            )
            async for task in utils.limited_as_completed_from_async_coro_gen(
                coro_gen, config.LOCAL_INDEX_BUILDING_NUM_THREADS
            ):
                await task

            self.logger.info("Local flat index: finished consuming Map output")
            await utils.run_in_executor(self.local_flat_index.build_distance_matrix)
            self.logger.info("Local flat index: finished building distance matrix")

    def start_training(
        self,
        mapper_result: MapperReducer.Result,
        index_type: Optional[IndexType] = None,
    ):
        if mapper_result.finished:
            self.logger.info(
                "Map: finished after successfully processing "
                f"{mapper_result.num_images} images"
            )

        # Step 3: "Train" each index once we have enough images/spatial embeddings (or
        # when the Map step finishes, in which case index_type=None indicating that we
        # should train all remaining indexes)
        index_types = [index_type] if index_type else iter(IndexType)
        for index_type in index_types:
            if self.training_jobs[index_type].started:
                continue
            self.training_jobs[index_type].start(mapper_result)
            self.logger.info(
                f"Train ({index_type.name}): started with "
                f"{mapper_result.num_embeddings} embeddings for "
                f"{mapper_result.num_images} images"
            )

    async def handle_training_status_update(self, result: JSONType):
        # Because training takes a long time, the trainer sends us back an HTTP request
        # on status changes rather than communicating over a single request/response.
        # This function is called by the Sanic endpoint and passes the status update
        # along to the relevant training job.
        index_type = IndexType[result["index_name"]]
        self.logger.debug(f"Train ({index_type.name}): recieved status update {result}")

        if index_type in self.training_jobs:
            await self.training_jobs[index_type].handle_result(result)

    @utils.unasync_as_task
    async def start_adding_eventually(self):
        self.index_dir.mkdir(parents=True, exist_ok=False)
        indexes = {}

        # Wait until all indexes are trained
        async for done in utils.as_completed_from_futures(
            [
                asyncio.create_task(job.finished.wait(), name=index_type.name)
                for index_type, job in self.training_jobs.items()
            ]
        ):
            assert isinstance(done, asyncio.Task)
            await done

            index_type = IndexType[done.get_name()]
            job = self.training_jobs[index_type]
            indexes[index_type.name] = {
                "reduction": "average" if job.average else None,
                "index_dir": job.index_dir,
            }
            self.logger.info(f"Train ({index_type.name}): finished")

            # Copy index training results to local disk before anything else gets
            # written into the index directory on the shared disk
            index_subdir = self.index_dir / index_type.name
            shutil.copytree(job.mounted_index_dir, index_subdir)
            self.logger.debug(
                f"Train ({index_type.name}): copied trained index from shared disk "
                f"({job.mounted_index_dir}) to local disk ({index_subdir})"
            )

        # TODO(mihirg): Fix skipping issue in Map and Add when started concurrently,
        # then remove this line!
        await self.mapper_job.reducer.finished.wait()

        # Step 4: As the Map step computes and saves embeddings, "Add" them into shards
        # of the newly trained indexes
        # TODO(mihirg): Consider adding to each index independently as training
        # finishes, then merging independently, then making indexes available on the
        # frontend for queries as they are completed
        nproc = self.cluster.output["adder_nproc"]
        n_mappers = int(
            self.cluster.output["num_adders"] * config.ADDER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.ADDER_CHUNK_SIZE(nproc)
        self.adder_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["adder_url"],
                n_mappers=n_mappers,
            ),
            AdderReducer(),
            {"indexes": indexes},
            session=self.http_session,
            n_retries=config.ADDER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.ADDER_REQUEST_TIMEOUT,
        )
        await self.adder_job.start(
            self.mapper_job.reducer.output_path_tmpl_gen(),
            self.start_resizing_and_merging,
        )  # iterable is an async generator that yields as the Map step produces outputs
        self.logger.info("Add: started")

    @utils.unasync_as_task
    async def start_resizing_and_merging(self, shard_patterns: Set[str]):
        self.logger.info(f"Add: finished with {len(shard_patterns)} shard patterns")
        self.merge_task = self.merge_indexes_in_background(shard_patterns)

        # Step 5: "Resize" images into small thumbnails so that the frontend can render
        # faster
        assert self.cluster  # just to silence type warnings
        iterable = (
            {"image": label, "identifier": identifier}
            for label, identifier in zip(self.labels, self.train_identifiers.keys())
        )
        nproc = self.cluster.output["resizer_nproc"]
        n_mappers = int(
            self.cluster.output["num_resizers"] * config.RESIZER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.RESIZER_CHUNK_SIZE(nproc)
        self.resizer_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["resizer_url"],
                n_mappers=n_mappers,
            ),
            IsFinishedReducer(),
            {
                "input_bucket": self.bucket,
                "output_bucket": config.RESIZER_OUTPUT_BUCKET,
                "output_dir": config.RESIZER_OUTPUT_DIR_TMPL.format(self.index_id),
                "resize_max_height": config.RESIZER_MAX_HEIGHT,
            },
            session=self.http_session,
            n_retries=config.RESIZER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.RESIZER_REQUEST_TIMEOUT,
        )
        await self.resizer_job.start(
            iterable,
            lambda _: self.logger.info("Resize: finished"),
        )
        self.logger.info("Resize: started")

    @utils.unasync_as_task
    async def merge_indexes_in_background(self, shard_patterns: Set[str]):
        loop = asyncio.get_running_loop()

        # Step 6: "Merge" shards from shared disk into final local index (in a thread
        # pool; because FAISS releases the GIL, this won't block the event loop)
        # TODO(mihirg): Consider deleting all unnecessary intermediates from NAS after
        self._load_local_indexes()
        with concurrent.futures.ThreadPoolExecutor(len(self.indexes)) as pool:
            futures = []
            for index_type, index in self.indexes.items():
                index_dir = self.training_jobs[index_type].mounted_index_dir
                future = asyncio.ensure_future(
                    loop.run_in_executor(
                        pool, self.merge_index, index, shard_patterns, index_dir
                    )
                )
                self.logger.info(f"Merge ({index_type.name}): started")
                future.add_done_callback(
                    lambda _, index_type=index_type: self.logger.info(
                        f"Merge ({index_type.name}): finished"
                    )
                )
                futures.append(future)

            await asyncio.gather(*futures)

        # Upload final index to Cloud Storage
        await self.build_local_flat_index_task
        await self.upload()

        # Wait for resizing to complete
        await self.resizer_job.reducer.finished.wait()

        self.logger.info("Finished building index")
        self.ready.set()
        self.cluster_unlock_fn()

    @staticmethod
    def merge_index(index: InteractiveIndex, shard_patterns: Set[str], index_dir: Path):
        shard_paths = [
            str(p.resolve())
            for shard_pattern in shard_patterns
            for p in index_dir.glob(shard_pattern)
        ]
        index.merge_partial_indexes(shard_paths)

    async def upload(self):
        # Dump labels, identifiers, full-image embeddings, and distance matrix
        json.dump(self.labels, (self.index_dir / self.LABELS_FILENAME).open("w"))
        json.dump(
            self.train_identifiers,
            (self.index_dir / self.TRAIN_IDENTIFIERS_FILENAME).open("w"),
        )
        json.dump(
            self.val_identifiers,
            (self.index_dir / self.VAL_IDENTIFIERS_FILENAME).open("w"),
        )
        self.local_flat_index.save(self.index_dir)

        # Upload to Cloud Storage
        # TODO(mihirg): Speed up
        # https://medium.com/google-cloud/google-cloud-storage-large-object-upload-speeds-7339751eaa24
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            str(self.index_dir),
            config.INDEX_UPLOAD_GCS_PATH,
        )
        await proc.wait()

    @property
    def status(self):
        return {
            "map": self.mapper_job.status if self.mapper_job else {},
            "add": self.adder_job.status if self.adder_job else {},
            "train": {
                index_type.name: job.status
                for index_type, job in self.training_jobs.items()
            },
            "resize": self.resizer_job.status if self.resizer_job else {},
        }

    # INDEX LOADING

    @classmethod
    async def load(
        cls,
        index_id: str,
        download: bool = False,
        *args,
        **kwargs,
    ) -> "LabeledIndex":
        self = cls(index_id, *args, **kwargs)

        if download:
            # Download from Cloud Storage
            config.INDEX_PARENT_DIR.mkdir(parents=True, exist_ok=True)
            # TODO(mihirg): Speed up
            # https://medium.com/@duhroach/gcs-read-performance-of-large-files-bd53cfca4410
            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                "-n",
                f"{config.INDEX_UPLOAD_GCS_PATH}{self.index_id}",
                str(config.INDEX_PARENT_DIR),
            )
            await proc.wait()

        # Initialize indexes
        self.labels = json.load((self.index_dir / self.LABELS_FILENAME).open())
        try:
            self.train_identifiers = json.load(
                (self.index_dir / self.TRAIN_IDENTIFIERS_FILENAME).open()
            )
            self.val_identifiers = json.load(
                (self.index_dir / self.VAL_IDENTIFIERS_FILENAME).open()
            )
            for model, dim in config.EMBEDDING_DIMS_BY_MODEL.items():
                try:
                    self.local_flat_indexes[model] = LocalFlatIndex.load(
                        self.index_dir / "local" / model, len(self.labels), dim
                    )
                except Exception as e:
                    self.logger.warning(f"Error loading local flat index from {self.index_dir}: {e}")
        except Exception as e:
            self.logger.warning(f"Error loading index from {self.index_dir}: {e}")
        try:
            self._load_local_indexes()
        except Exception as e:
            self.logger.warning(f"Error loading FAISS indexes: {e}")
        self.logger.info(f"Finished loading index from {self.index_dir}")

        self.ready.set()
        return self

    def _load_local_indexes(self):
        self.indexes = {
            index_type: InteractiveIndex.load(str(self.index_dir / index_type.name))
            for index_type in IndexType
        }


# Start web server
app = Sanic(__name__)
app.update_config({"RESPONSE_TIMEOUT": config.SANIC_RESPONSE_TIMEOUT})


# CLUSTER


async def _start_cluster(cluster):
    # Create cluster
    # Hack(mihirg): just attach mounting-related attributes to the cluster object
    cluster.mounted = asyncio.Event()
    await cluster.apply()

    # Mount NFS
    cluster.mount_parent_dir = config.CLUSTER_MOUNT_DIR / cluster.id
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
    if not config.CLUSTER_REUSE_EXISTING:
        await cluster.destroy()


# TODO(mihirg): Automatically clean up inactive clusters
current_clusters: CleanupDict[str, TerraformModule] = CleanupDict(
    _stop_cluster, app.add_task, config.CLUSTER_CLEANUP_TIME
)


@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    cluster = TerraformModule(
        config.CLUSTER_TERRAFORM_MODULE_PATH, copy=not config.CLUSTER_REUSE_EXISTING
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


# INDEX


current_indexes: CleanupDict[str, LabeledIndex] = CleanupDict(
    lambda job: job.stop_building()
)

# -> BUILD
# TODO(mihirg): Consider restructuring to simplfiy


@app.route("/start_job", methods=["POST"])
async def start_job(request):
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    paths = request.json["paths"]
    identifiers = request.json["identifiers"]

    cluster = current_clusters[cluster_id]
    lock_id = current_clusters.lock(cluster_id)
    cluster_unlock_fn = functools.partial(current_clusters.unlock, cluster_id, lock_id)
    index = await LabeledIndex.start_building(
        cluster, cluster_unlock_fn, bucket, paths, identifiers
    )

    index_id = index.index_id
    current_indexes[index_id] = index
    return resp.json({"index_id": index_id})


@app.route(config.TRAINER_STATUS_ENDPOINT, methods=["PUT"])
async def training_status(request):
    index_id = request.json["index_id"]
    if index_id in current_indexes:
        await current_indexes[index_id].handle_training_status_update(request.json)
    return resp.text("", status=204)


@app.route("/job_status", methods=["GET"])
async def job_status(request):
    index_id = request.args["index_id"][0]
    if index_id in current_indexes:
        index = current_indexes[index_id]
        status = index.status
        status["has_index"] = index.ready.is_set()
    else:
        status = {"has_index": False}
    return resp.json(status)


# TODO(mihirg): Do we even need this function? It's not exposed on the frontend.
@app.route("/stop_job", methods=["POST"])
async def stop_job(request):
    index_id = request.json["index_id"]
    app.add_task(current_indexes.cleanup_key(index_id))
    return resp.text("", status=204)


# -> BGSPLIT-TRAIN

current_models: CleanupDict[str, BGSplitTrainingJob] = CleanupDict(
    lambda job: job.stop()
)


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
    gcs_root_path = os.path.join(config.GCS_PUBLIC_ROOT_URL, bucket)
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
    aux_labels_gcs_path = config.AUX_GCS_PUBLIC_TMPL.format(index_id, alt)

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


@app.route(config.BGSPLIT_TRAINER_STATUS_ENDPOINT, methods=["PUT"])
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


current_model_inference_jobs: CleanupDict[str, BGSplitInferenceJob] = CleanupDict(
    lambda job: job.stop()
)


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
    gcs_root_path = os.path.join(config.GCS_PUBLIC_ROOT_URL, bucket)
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


@app.route("/stop_bgsplit_inference_job", methods=["POST"])
async def bgsplit_inference_job_status(request):
    job_id = request.args["job_id"][0]
    if job_id in current_model_inference_jobs:
        job = current_model_inference_jobs[job_id]
        job.stop()
        del current_model_inference_jobs[job_id]
        status["has_output"] = False
        status["finished"] = True
        status["failed"] = False
        return resp.json(status)
    else:
        status = {"reason": f"Job id {job_id} does not exist."}
        return resp.json(status, status=400)


# -> POST-BUILD


async def _download_index(index_id):
    if index_id not in current_indexes:
        current_indexes[index_id] = await LabeledIndex.load(index_id, download=True)


@app.route("/download_index", methods=["POST"])
async def download_index(request):
    index_id = request.json["index_id"]
    app.add_task(_download_index(index_id))
    return resp.text("", status=204)


# TODO(mihirg): Do we even need this function?
@app.route("/delete_index", methods=["POST"])
async def delete_index(request):
    index_id = request.json["index_id"]
    await current_indexes.pop(index_id).delete()
    return resp.text("", status=204)


# QUERY
# TODO(mihirg): Change input from form encoding to JSON
# TODO(all): Clean up this code


async def get_index(index_id) -> LabeledIndex:
    if index_id not in current_indexes:
        current_indexes[index_id] = await LabeledIndex.load(index_id)
    return current_indexes[index_id]


def extract_embedding_from_mapper_output(output: str) -> np.ndarray:
    return np.squeeze(utils.base64_to_numpy(output), axis=0)


class BestMapper:
    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.lock_id: Optional[str] = None

    async def __aenter__(self) -> MapperSpec:  # returns endpoint
        cluster = current_clusters.get(self.cluster_id)
        if cluster and cluster.ready.is_set():
            self.lock_id = current_clusters.lock(self.cluster_id)

            nproc = cluster.output["mapper_nproc"]
            n_mappers = int(
                cluster.output["num_mappers"] * config.MAPPER_REQUEST_MULTIPLE(nproc)
            )
            return MapperSpec(url=cluster.output["mapper_url"], n_mappers=n_mappers)
        else:
            return MapperSpec(
                url=config.MAPPER_CLOUD_RUN_URL, n_mappers=config.CLOUD_RUN_N_MAPPERS
            )

    async def __aexit__(self, type, value, traceback):
        if self.lock_id:
            current_clusters.unlock(self.cluster_id, self.lock_id)


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    image_paths = request.json["paths"]
    identifiers = request.json["identifiers"]
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in request.json["patches"]
    ]  # [0, 1]^2
    index_id = request.json["index_id"]
    num_results = int(request.json["num_results"])
    augmentations = request.json.get("augmentations", [])

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = bool(request.json.get("use_full_image", False))

    index = await get_index(index_id)

    # Generate query vector as average of patch embeddings
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            VectorReducer(
                VectorReducer.PoolingType.AVG,
                extract_func=extract_embedding_from_mapper_output,
            ),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        query_vector = await job.run_until_complete(
            [
                {
                    "image": image_path,
                    "patch": patch,
                    "augmentations": augmentation_dict,
                }
                for image_path, patch in zip(image_paths, patches)
            ]
        )

    # Run query and return results
    query_results = index.query(query_vector, num_results, None, use_full_image, False)
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/active_batch", methods=["POST"])
async def active_batch(request):
    image_paths = request.json["paths"]  # Paths of seed images (at first from google)
    bucket = request.json["bucket"]
    cluster_id = request.json["cluster_id"]
    index_id = request.json["index_id"]
    num_results = int(request.json["num_results"])
    augmentations = request.json.get("augmentations", [])

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = True

    index = await get_index(index_id)

    # Generate query vector as average of patch embeddings
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            VectorReducer(extract_func=extract_embedding_from_mapper_output),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        query_vectors = await job.run_until_complete(
            [
                {"image": image_path, "augmentations": augmentation_dict}
                for image_path in image_paths
            ]
        )

    all_results = {}
    # Results per vector
    perVector = (int)(num_results / len(query_vectors)) + 2
    for vec in query_vectors:
        # Return nearby images
        query_results = index.query(
            np.float32(vec),
            perVector,
            None,
            use_full_image,
            False,  # False bc just standard nearest neighbor
        )

        # Remove duplicates; for each image, include closest result
        for result in query_results:
            label = result.label
            if label not in all_results or all_results[label].dist > result.dist:
                all_results[label] = result

    return resp.json(
        {"results": [r.to_dict() for r in all_results.values()]}
    )  # unordered by distance for now


# TODO(mihirg): Reuse these instances (maybe in an ExpiringDict, storing embeddings in
# a Chest internally?) to cache embeddings and speed up repeatedly iterating on training
# an SVM on a particular dataset
class SVMExampleReducer(Reducer):
    def __init__(self):
        self.labels: List[int] = []
        self.embeddings: List[np.ndarray] = []

    def handle_result(self, input: JSONType, output: str):
        label = int(bool(input["label"]))
        self.labels.append(label)
        self.embeddings.append(extract_embedding_from_mapper_output(output))

    @property
    def result(self) -> Tuple[np.ndarray, np.ndarray]:  # features, labels
        return np.stack(self.embeddings), np.array(self.labels)


@app.route("/query_svm", methods=["POST"])
async def query_svm(request):
    index_id = request.json["index_id"]
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    pos_image_paths = request.json["positive_paths"]
    pos_patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in request.json["positive_patches"]
    ]  # [0, 1]^2
    neg_image_paths = request.json["negative_paths"]
    num_results = int(request.json["num_results"])
    mode = request.json["mode"]
    use_full_image = bool(request.json.get("use_full_image", False))

    # Automatically label `autolabel_max_vectors` vectors randomly sampled from the
    # bottom `autolabel_percent`% of the previous SVM's results as negative
    index = await get_index(index_id)

    prev_svm_vector = utils.base64_to_numpy(request.json["prev_svm_vector"])
    autolabel_percent = float(request.json["autolabel_percent"])
    autolabel_max_vectors = int(request.json["autolabel_max_vectors"])
    log_id_string = request.json["log_id_string"]

    if (
        prev_svm_vector is not None
        and autolabel_percent > 0
        and autolabel_max_vectors > 0
    ):
        already_labeled_image_paths = set(
            itertools.chain(pos_image_paths, neg_image_paths)
        )
        autolabel_results = index.query_farthest(
            prev_svm_vector,
            autolabel_percent / 100,  # percentage to fraction!
            autolabel_max_vectors,
            use_full_image,
        )
        autolabel_image_paths = [
            r.label
            for r in autolabel_results
            if r.label not in already_labeled_image_paths
        ]
    else:
        autolabel_image_paths = []

    # Generate training vectors
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            SVMExampleReducer(),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        pos_inputs = [
            {"image": image_path, "patch": patch, "label": 1}
            for image_path, patch in zip(pos_image_paths, pos_patches)
        ]
        neg_inputs = [
            {"image": image_path, "label": 0} for image_path in neg_image_paths
        ]
        auto_inputs = [
            {"image": image_path, "label": 0} for image_path in autolabel_image_paths
        ]

        logger.info(
            f"{log_id_string} - Starting SVM training vector computation: {len(pos_inputs)} positives, "
            f"{len(neg_inputs)} negatives, {len(auto_inputs)} auto-negatives"
        )
        training_features, training_labels = await job.run_until_complete(
            itertools.chain(pos_inputs, neg_inputs, auto_inputs)
        )
        logger.info(
            f"{log_id_string} - Finished SVM vector computation in {job.elapsed_time:.3f}s"
        )
        logger.debug(
            f"{log_id_string} - Vector computation performance: {job.performance}"
        )

    # Train SVM
    logger.debug(f"{log_id_string} - Starting SVM training")
    start_time = time.perf_counter()

    model = svm.SVC(kernel="linear")
    model.fit(training_features, training_labels)
    predicted = model.predict(training_features)

    end_time = time.perf_counter()
    logger.info(
        f"{log_id_string} - Finished training SVM in {end_time - start_time:.3f}s"
    )
    logger.debug(
        f"{log_id_string} - SVM accuracy: {accuracy_score(training_labels, predicted)}"
    )

    if mode == "svmPos" or mode == "spatialSvmPos":
        # Evaluate the SVM by querying index
        w = model.coef_  # This will be the query vector
        # Also consider returning the support vectors; good to look at examples along
        # hyperplane
        w = np.float32(w[0] * 1000)

        augmentations = request.json.get("augmentations", [])

        augmentation_dict = {}
        for i in range(len(augmentations) // 2):
            augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

        # Run query and return results
        query_results = index.query(w, num_results, None, use_full_image, True)
        return resp.json(
            {
                "results": [r.to_dict() for r in query_results],
                "svm_vector": utils.numpy_to_base64(w),
            }
        )
    elif mode == "svmBoundary":
        # Get samples close to boundary vectors
        # For now, looks like most vectors end up being support vectors since
        # underparamtrized system
        sv = model.support_vectors_
        all_results = {}
        # Results per vector
        perVector = (int)(num_results / len(sv)) + 2
        for vec in sv:
            # Return nearby images
            query_results = index.query(
                np.float32(vec),
                perVector,
                None,
                use_full_image,
                False,  # False bc just standard nearest neighbor
            )

            # Remove duplicates; for each image, include closest result
            for result in query_results:
                label = result.label
                if label not in all_results or all_results[label].dist > result.dist:
                    all_results[label] = result

        return resp.json(
            {
                "results": [r.to_dict() for r in all_results.values()],
                "svm_vector": utils.numpy_to_base64(w),
            }
        )  # unordered by distance for now
    else:
        return resp.json({"results": []})


# NEW FRONTEND


@app.route("/perform_clustering", methods=["POST"])
async def perform_clustering(request):
    identifiers = request.json["identifiers"]
    index_id = request.json["index_id"]
    args = dict(identifiers=identifiers)
    if "model" in request.json:
        args["model"] = request.json["model"]
    index = await get_index(index_id)
    clustering = index.cluster_identifiers(**args)
    return resp.json({"clustering": clustering})


@app.route("generate_embedding", methods=["POST"])
async def generate_embedding(request):
    identifier = request.json.get("identifier")
    if identifier:
        index_id = request.json["index_id"]
        index = await get_index(index_id)
        embedding = index.get_embeddings([identifier])[0]
    else:
        image_data = re.sub("^data:image/.+;base64,", "", request.json["image_data"])

        # Upload to Cloud Storage
        async with aiohttp.ClientSession() as session:
            storage_client = Storage(session=session)
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            bucket = config.UPLOADED_IMAGE_BUCKET
            path = os.path.join(config.UPLOADED_IMAGE_DIR, f"{uuid.uuid4()}.png")
            with BytesIO() as image_buffer:
                image.save(image_buffer, "jpeg")
                image_buffer.seek(0)
                await storage_client.upload(bucket, path, image_buffer)

            # Compute embedding using Mapper
            mapper = MapperSpec(
                url=config.MAPPER_CLOUD_RUN_URL, n_mappers=config.CLOUD_RUN_N_MAPPERS
            )
            job = MapReduceJob(
                mapper,
                VectorReducer(
                    VectorReducer.PoolingType.AVG,
                    extract_func=extract_embedding_from_mapper_output,
                ),
                {"input_bucket": bucket, "reduction": "average"},
                n_retries=config.CLOUD_RUN_N_RETRIES,
                chunk_size=1,
                session=session,
            )
            embedding = await job.run_until_complete([{"image": path}])

    return resp.json({"embedding": utils.numpy_to_base64(embedding)})


@app.route("generate_text_embedding", methods=["POST"])
async def generate_text_embedding(request):
    text = request.json["text"]
    mapper = MapperSpec(
        url=config.CLIP_TEXT_INFERENCE_CLOUD_RUN_URL,
        n_mappers=config.CLOUD_RUN_N_MAPPERS,
    )
    job = MapReduceJob(
        mapper,
        ListReducer(),
        n_retries=config.CLOUD_RUN_N_RETRIES,
        chunk_size=1,
    )
    embedding_base64 = (await job.run_until_complete([text]))[0]

    return resp.json({"embedding": embedding_base64})


@app.route("/query_knn_v2", methods=["POST"])
async def query_knn_v2(request):
    embeddings = request.json["embeddings"]
    index_id = request.json["index_id"]
    model = request.json["model"]
    use_dot_product = request.json["use_dot_product"]
    use_full_image = request.json["use_full_image"]
    assert use_full_image

    index = await get_index(index_id)

    # Get query vector from local flat index
    query_vector = np.mean([utils.base64_to_numpy(e) for e in embeddings], axis=0)

    # Run query and return results
    query_results = index.query_brute_force(
        query_vector, dot_product=use_dot_product, model=model
    )
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/train_svm_v2", methods=["POST"])
async def train_svm_v2(request):
    # HACK(mihirg): sometimes we get empty identifiers (i.e., "") from the server that
    # would otherwise cause a crash here; we should probably figure out why this is, but
    # just filtering out for now.
    pos_identifiers = list(filter(bool, request.json["pos_identifiers"]))
    neg_identifiers = list(filter(bool, request.json["neg_identifiers"]))
    index_id = request.json["index_id"]
    embedding_model = request.json["model"]

    index = await get_index(index_id)

    # Get positive and negative image embeddings from local flat index
    pos_vectors = index.get_embeddings(pos_identifiers, embedding_model)
    neg_vectors = index.get_embeddings(neg_identifiers, embedding_model)
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


@app.route("/query_svm_v2", methods=["POST"])
async def query_svm_v2(request):
    score_min = float(request.json["score_min"])
    score_max = float(request.json["score_max"])
    svm_vector = utils.base64_to_numpy(request.json["svm_vector"])
    index_id = request.json["index_id"]
    model = request.json["model"]

    index = await get_index(index_id)

    # Run query and return results
    query_results = index.query_brute_force(
        svm_vector, dot_product=True, min_d=score_min, max_d=score_max, model=model
    )
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/query_ranking_v2", methods=["POST"])
async def query_ranking_v2(request):
    index_id = request.json["index_id"]
    model = request.json["model"]
    score_min = float(request.json["score_min"])
    score_max = float(request.json["score_max"])

    index = await get_index(index_id)

    # Run query and return results
    query_results = index.rank_brute_force(model, score_min, score_max)
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/query_metrics", methods=["POST"])
async def query_metrics(request):
    index_id = request.json["index_id"]
    model = request.json["model"]
    identifiers = request.json["identifiers"]  # type: List[str]
    labels = request.json["labels"]  # type: List[bool]
    weights = request.json["weights"]  # type: List[float]

    assert len(identifiers) == len(labels) == len(weights)
    num_labeled = len(identifiers)

    index = await get_index(index_id)

    prob_pos = index.get_model_scores(model, identifiers)
    y_pred = prob_pos > config.DNN_SCORE_CLASSIFICATION_THRESHOLD
    y_test = np.array(labels)
    rows = np.arange(num_labeled)
    weights = np.array(weights)

    precision, precision_std, _ = ais.get_fscore(y_pred, y_test, rows, weights * y_pred)
    recall, recall_std, _ = ais.get_fscore(y_pred, y_test, rows, weights * y_test)
    f1, f1_std, _ = ais.get_fscore(
        y_pred, y_test, rows, weights * (0.5 * y_pred + 0.5 * y_test)
    )

    false_positives = []
    false_negatives = []
    for identifier, label, score in zip(identifiers, labels, prob_pos):
        result = LabeledIndex.QueryResult(
            index.val_identifiers[identifier], float(score)
        )
        if score > config.DNN_SCORE_CLASSIFICATION_THRESHOLD and not label:
            false_positives.append(result)
        elif score <= config.DNN_SCORE_CLASSIFICATION_THRESHOLD and label:
            false_negatives.append(result)

    false_positives.sort(key=operator.attrgetter("dist"), reverse=True)  # descending
    false_negatives.sort(key=operator.attrgetter("dist"))  # ascending

    for result in itertools.chain(false_positives, false_negatives):
        result.label = index.labels[result.id]

    results = {
        "precision": precision,
        "precision_std": precision_std,
        "recall": recall,
        "recall_std": recall_std,
        "f1": f1,
        "f1_std": f1_std,
        # "false_positives": [r.to_dict() for r in false_positives],
        # "false_negatives": [r.to_dict() for r in false_negatives],
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
    index_id = request.json["index_id"]
    model = request.json["model"]
    identifiers = request.json["identifiers"]  # type: List[str]
    labels = request.json["labels"]  # type: List[bool]
    current_f1 = float(request.json["current_f1"])

    assert len(identifiers) == len(labels)

    index = await get_index(index_id)

    all_val_identifiers = index.get_val_identifiers()
    prob_pos = index.get_model_scores(model, all_val_identifiers)
    y_pred = prob_pos > config.DNN_SCORE_CLASSIFICATION_THRESHOLD
    sample_budget = max(2 * len(labels), config.ACTIVE_VAL_STARTING_BUDGET)
    g = current_f1
    alpha = 0.5

    val_identifiers_to_inds = {id: i for i, id in enumerate(all_val_identifiers)}
    known_rows_inds = [val_identifiers_to_inds[id] for id in identifiers]

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
    rows, weights = ais.ais_singleiter(
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
    identifiers_to_weights = {}  # type: Dict[str, float]
    for index, weight in zip(rows, weights):
        id = all_val_identifiers[index]
        identifiers_to_weights[id] = weight
        if id not in old_identifiers:
            new_identifiers.append(id)

    return resp.json(
        {
            "identifiers": new_identifiers,
            "weights": identifiers_to_weights,
        }
    )


last_keep_alive_time = 0


async def _keep_alive():
    global last_keep_alive_time
    if time.time() - last_keep_alive_time < config.MIN_TIME_BETWEEN_KEEP_ALIVES:
        return
    last_keep_alive_time = time.time()

    async def keep_endpoint_alive(session, endpoint):
        async with session.post(endpoint):
            pass

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            keep_endpoint_alive(session, config.MAPPER_CLOUD_RUN_URL),
            keep_endpoint_alive(session, config.CLIP_TEXT_INFERENCE_CLOUD_RUN_URL),
        )


@app.route("/keep_alive", methods=["POST"])
async def keep_alive(request):
    app.add_task(_keep_alive())
    return resp.text("", status=204)


# CLEANUP


@app.listener("after_server_stop")
async def cleanup(app, loop):
    print("Terminating:")
    await _cleanup_indexes()
    await _cleanup_clusters()


@utils.log_exception_from_coro_but_return_none
async def _cleanup_indexes():
    n = len(current_indexes)
    await current_indexes.clear_async()
    print(f"- cleaned up {n} indexes")


@utils.log_exception_from_coro_but_return_none
async def _cleanup_clusters():
    n = len(current_clusters)
    await current_clusters.clear_async()
    print(f"- killed {n} clusters")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
