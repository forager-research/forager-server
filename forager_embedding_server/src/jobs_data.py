import functools
import logging
import operator
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import fastcluster
import numpy as np
from dataclasses_json import dataclass_json

from forager_embedding_server.config import CONFIG
from forager_embedding_server.utils import sha_encode


@functools.lru_cache(maxsize=32)
def load_image_list(path: str):
    return ImageList(path)


@functools.lru_cache(maxsize=32)
def load_score_set(path: str, image_list_path: str):
    images = load_image_list(image_list_path)
    return ScoreSet(path, images)


@functools.lru_cache(maxsize=32)
def load_embedding_set(path: str, image_list_path: str, dtype=np.float32):
    images = load_image_list(image_list_path)
    return EmbeddingSet(path, images, dtype)


@dataclass_json
@dataclass
class QueryResult:
    id: int
    dist: float = 0.0
    identifier: str = ""


class ImageList:
    def __init__(self, path: str):
        self.identifiers_to_inds = {}
        self.inds_to_identifiers = {}
        self.inds_to_paths = {}
        self.splits_to_ind_sets = defaultdict(set)
        with open(path) as f:
            for ind, line in enumerate(f):
                sep1 = line.find(" ")
                sep2 = sep1 + 1 + line[sep1 + 1 :].find(" ")
                split = line[:sep1]
                identifier = line[sep1 + 1 : sep2]
                path = line[sep2 + 1 :].strip()

                self.identifiers_to_inds[identifier] = ind
                self.inds_to_identifiers[ind] = identifier
                self.inds_to_paths[ind] = path
                self.splits_to_ind_sets[split].add(ind)

    def __len__(self) -> int:
        return len(self.identifiers_to_inds)

    @staticmethod
    def write_from_image_paths(
        splits_to_image_paths: Dict[str, List[Tuple[str, str]]], f
    ):
        for split, paths in splits_to_image_paths.items():
            for path, ident in paths:
                f.write(f"{split} {ident} {path}\n")

    def get_ind(self, identifier: str) -> int:
        return self.identifiers_to_inds[identifier]

    def get_inds(self, identifiers: Iterable[str]) -> List[int]:
        return list(map(self.get_ind, identifiers))

    def get_identifier(self, ind: int) -> str:
        return self.inds_to_identifiers[ind]

    def get_identifiers(self, inds: Iterable[int]) -> List[str]:
        return list(map(self.get_identifier, inds))

    def get_path(self, ind: int) -> str:
        return self.inds_to_paths[ind]

    def get_paths(self, inds: Iterable[int]) -> List[str]:
        return list(map(self.get_path, inds))

    def get_inds_for_split(self, split: str) -> List[int]:
        return list(self.splits_to_ind_sets[split])


class ScoreSet:
    def __init__(self, path: str, images: ImageList):
        self.scores = np.load(path)
        self.images = images
        self.logger = logging.getLogger(
            f"index_server.ScoreSet({sha_encode(path)[:6]})"
        )

    # NOTE(mihirg): We don't normalize scores here because we expect model outputs in
    # [0, 1] anyway; consider adding as a param in the future
    def rank_brute_force(
        self, min_s: float = 0.0, max_s: float = 1.0
    ) -> List[QueryResult]:
        start = time.perf_counter()

        ranking = np.argsort(self.scores)[::-1]
        sorted_results = []
        for i in ranking:
            i = int(i)
            s = float(self.scores[i])
            if min_s <= s <= max_s:
                sorted_results.append(QueryResult(i, s, self.images.get_identifier(i)))

        end = time.perf_counter()
        self.logger.debug(
            f"Ranking query on {len(self.images)} vectors with score range ({min_s}, "
            f"{max_s}) took {end-start:.3f}s and found {len(sorted_results)} results."
        )

        return sorted_results

    def get_scores(
        self, identifiers: Optional[List[str]] = None, inds: Optional[List[int]] = None
    ) -> np.ndarray:
        if identifiers is None and inds is None:
            return self.scores
        if inds is None:
            inds = self.images.get_inds(identifiers)  # type: ignore
        return self.scores[inds]


class EmbeddingSet:
    def __init__(self, path: str, images: ImageList, dtype=np.float32):
        dim = int(os.path.getsize(path) / int(np.dtype(dtype).itemsize) / len(images))
        self.embeddings = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            shape=(len(images), dim),
        )
        self.images = images
        self.logger = logging.getLogger(
            f"index_server.EmbeddingSet({sha_encode(path)[:6]})"
        )

    def query_brute_force(
        self,
        query_vector: np.ndarray,
        dot_product: bool = False,
        min_d: float = 0.0,
        max_d: float = 1.0,
        chunk_size: int = CONFIG.BRUTE_FORCE_QUERY_CHUNK_SIZE,  # unused
    ) -> List[QueryResult]:
        start = time.perf_counter()

        # TODO(mihirg): Process CHUNK_SIZE rows at a time for large datasets
        if dot_product:
            dists = self.embeddings @ query_vector
        else:
            dists = cdist(np.expand_dims(query_vector, axis=0), self.embeddings)
            dists = np.squeeze(dists, axis=0)

        sorted_results = []
        lowest_dist = np.min(dists)
        highest_dist = np.max(dists)

        for i, d in enumerate(dists):
            d = float(d)
            d = (d - lowest_dist) / (highest_dist - lowest_dist)  # normalize
            if min_d <= d <= max_d:
                sorted_results.append(QueryResult(i, d, self.images.get_identifier(i)))

        sorted_results.sort(key=operator.attrgetter("dist"), reverse=dot_product)

        end = time.perf_counter()
        self.logger.debug(
            f"Search query on {len(self.images)} vectors (n_dim={len(query_vector)}, "
            f"dot_product={dot_product}) with distance range ({min_d}, {max_d}) took "
            f"{end-start:.3f}s and found {len(sorted_results)} results."
        )

        return sorted_results

    def get_embeddings(
        self, identifiers: List[str] = None, inds: Optional[List[int]] = None
    ) -> np.ndarray:
        if identifiers is None and inds is None:
            return self.embeddings
        if inds is None:
            inds = self.images.get_inds(identifiers)  # type: ignore
        return self.embeddings[inds]

    def cluster_identifiers(self, identifiers: List[str]) -> List[List[float]]:
        embeddings = self.get_embeddings(identifiers)
        return self._cluster(embeddings)

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
