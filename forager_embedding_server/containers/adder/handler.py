from collections import defaultdict

import numpy as np

from typing import Dict, List, Optional, Tuple

from interactive_index import InteractiveIndex

from knn import utils
from knn.mappers import Mapper
from knn.utils import JSONType

import config


class IndexBuildingMapper(Mapper):
    def initialize_container(self):
        self.shard_pattern_for_glob = config.SHARD_INDEX_NAME_TMPL.format(
            self.worker_id
        ).format("*")

    async def initialize_job(self, job_args) -> InteractiveIndex:
        index_dicts = job_args["indexes"]

        job_args["indexes_by_reduction"] = defaultdict(dict)
        for index_name, index_dict in index_dicts.items():
            reduction = index_dict["reduction"]
            index_dir = index_dict["index_dir"]

            index = InteractiveIndex.load(index_dir)
            index.SHARD_INDEX_NAME_TMPL = config.SHARD_INDEX_NAME_TMPL.format(
                self.worker_id
            )
            job_args["indexes_by_reduction"][reduction][index_name] = index

        return job_args

    # inputs = paths to np.save'd Dict[int, np.ndarray] where each value is N x D
    @utils.log_exception_from_coro_but_return_none
    async def process_chunk(
        self, chunk: List[str], job_id, job_args, request_id
    ) -> Optional[bool]:
        indexes_by_reduction = job_args["indexes_by_reduction"]
        path_tmpls = chunk

        for reduction, indexes in indexes_by_reduction.items():
            # Step 1: Load saved embeddings of entire chunk into memory
            with self.profiler(request_id, f"{reduction}_load_time_chunk"):
                embedding_dicts = [
                    np.load(path_tmpl.format(reduction), allow_pickle=True).item()
                    for path_tmpl in path_tmpls
                ]  # type: List[Dict[int, np.ndarray]]

            # Step 2: Extract embeddings
            with self.profiler(request_id, f"{reduction}_extract_time_chunk"):
                all_embeddings = np.concatenate(
                    [
                        embeddings
                        for embedding_dict in embedding_dicts
                        for embeddings in embedding_dict.values()
                    ]
                )
                all_image_ids = [
                    int(id)
                    for embedding_dict in embedding_dicts
                    for id, embeddings in embedding_dict.items()
                    for _ in range(embeddings.shape[0])
                ]
                all_spatial_ids = [
                    i
                    for embedding_dict in embedding_dicts
                    for embeddings in embedding_dict.values()
                    for i in range(embeddings.shape[0])
                ]

            # Step 3: Add to applicable on-disk indexes
            for index_name, index in indexes.items():
                with self.profiler(request_id, f"{index_name}_add_time_chunk"):
                    index.add(
                        all_embeddings,
                        all_image_ids,
                        ids_extra=all_spatial_ids,
                        update_metadata=False,
                    )

        return True  # success

    async def process_element(self, *args, **kwargs):
        raise NotImplementedError()

    async def postprocess_chunk(
        self,
        inputs,
        outputs: JSONType,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return self.shard_pattern_for_glob, [outputs] * len(inputs)


app = IndexBuildingMapper().server
