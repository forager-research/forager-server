import asyncio
import collections
import resource
import time
import textwrap
import traceback
import uuid

import aiohttp
from aiostream import stream
from runstats import Statistics

from forager_knn import utils
from forager_knn.clusters import GKECluster
from forager_knn.utils import JSONType
from forager_knn.reducers import Reducer

from . import defaults

from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)


# Increase maximum number of open sockets if necessary
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = max(min(defaults.DESIRED_ULIMIT, hard), soft)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

InputSequenceType = Union[Iterable[JSONType], AsyncIterable[JSONType]]


class MapperSpec:
    def __init__(
        self,
        *,
        url: Optional[str] = None,
        container: Optional[str] = None,
        cluster: Optional[GKECluster] = None,
        n_mappers: Optional[int] = None,
    ):
        assert (all((url, n_mappers)) and not any((container, cluster))) or (
            not url and all((container, cluster))
        )
        n_mappers = n_mappers or defaults.N_MAPPERS
        assert n_mappers < new_soft

        self.url = url
        self.container = container
        self.cluster = cluster

        self.n_mappers = n_mappers

        # Will be initialized on enter
        self._deployment_id: Optional[str] = None

    async def __aenter__(self) -> str:  # returns endpoint
        if self.url:
            return self.url
        else:
            # Start Kubernetes service
            assert self.cluster and self.container
            self._deployment_id, url = await self.cluster.create_deployment(
                self.container, self.n_mappers
            )
            return url

    async def __aexit__(self, type, value, traceback):
        if not self.url:
            # Stop Kubernetes service
            await self.cluster.delete_deployment(self._deployment_id)


class MapReduceJob:
    def __init__(
        self,
        mapper: MapperSpec,
        reducer: Reducer,
        mapper_args: JSONType = {},
        *,
        session: Optional[aiohttp.ClientSession] = None,
        n_retries: int = defaults.N_RETRIES,
        chunk_size: int = defaults.CHUNK_SIZE,
        request_timeout: int = defaults.REQUEST_TIMEOUT,
    ) -> None:
        self.job_id = str(uuid.uuid4())

        self.mapper = mapper
        self.mapper_args = mapper_args

        self.reducer = reducer

        self.n_retries = n_retries
        self.chunk_size = chunk_size
        self.request_timeout = aiohttp.ClientTimeout(total=request_timeout)

        self.owns_session = session is None
        self.session = session or utils.create_unlimited_aiohttp_session()

        # Performance stats
        self._n_requests = 0
        self._n_successful = 0
        self._n_failed = 0
        self._n_chunks_per_mapper: Dict[str, int] = collections.defaultdict(int)
        self._profiling: Dict[str, Statistics] = collections.defaultdict(Statistics)

        # Will be initialized later
        self._n_total: Optional[int] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    # REQUEST LIFECYCLE

    async def start(
        self,
        iterable: InputSequenceType,
        callback: Optional[Callable[[Any], None]] = None,
        n_total: Optional[int] = None,
    ) -> None:
        async def task():
            try:
                result = await self.run_until_complete(iterable, n_total)
            except asyncio.CancelledError:
                pass
            else:
                if callback:
                    callback(result)

        self._task = asyncio.create_task(task())

    async def run_until_complete(
        self,
        iterable: InputSequenceType,
        n_total: Optional[int] = None,
    ) -> Any:
        try:
            self._n_total = n_total or len(iterable)  # type: ignore
        except Exception:
            pass

        assert self._start_time is None  # can't reuse Job instances
        self._start_time = time.time()

        try:
            chunk_stream = stream.chunks(stream.iterate(iterable), self.chunk_size)

            async with self.mapper as mapper_url, chunk_stream.stream() as chunk_gen:
                async for response in utils.limited_as_completed_from_async_coro_gen(
                    (self._request(mapper_url, chunk) async for chunk in chunk_gen),
                    self.mapper.n_mappers,
                ):
                    response_tuple = await response
                    self._reduce_chunk(*response_tuple)

            if self._n_total is None:
                self._n_total = self._n_successful + self._n_failed
            else:
                assert self._n_total == self._n_successful + self._n_failed

            self.reducer.finish()
            return self.result
        finally:
            self._end_time = time.time()

            if self.owns_session:
                await self.session.close()

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await self._task

    # RESULT GETTERS

    @property
    def result(self) -> Any:
        return self.reducer.result

    @property
    def progress(self) -> Any:
        progress = {
            "cost": self.cost,
            "finished": self.finished,
            "n_processed": self._n_successful,
            "n_skipped": self._n_failed,
            "elapsed_time": self.elapsed_time,
        }
        if self._n_total is not None:
            progress["n_total"] = self._n_total

        return progress

    @property
    def elapsed_time(self):
        end_time = self._end_time or time.time()
        start_time = self._start_time or end_time
        return end_time - start_time

    @property
    def performance(self):
        return {
            "profiling": {
                k: {
                    "mean": v.mean(),
                    "std": v.stddev() if len(v) > 1 else 0,
                    "n": len(v),
                }
                for k, v in self._profiling.items()
            },
            "mapper_utilization": dict(enumerate(
                self._n_chunks_per_mapper.values())),
        }

    @property
    def status(self) -> Dict[str, Any]:
        return {"performance": self.performance, "progress": self.progress}

    @property
    def finished(self) -> bool:
        return self._n_total == self._n_successful + self._n_failed

    @property
    def cost(self) -> float:
        return 0  # not implemented for GKE

    # INTERNAL

    def _construct_request(self, chunk: List[JSONType]) -> JSONType:
        return {
            "job_id": self.job_id,
            "job_args": self.mapper_args,
            "inputs": chunk,
        }

    async def _request(
        self, mapper_url: str, chunk: List[JSONType]
    ) -> Tuple[JSONType, Optional[JSONType], float]:
        result = None
        start_time = 0.0
        end_time = 0.0

        request = self._construct_request(chunk)

        for i in range(self.n_retries):
            start_time = time.time()
            end_time = start_time

            try:
                async with self.session.post(
                    mapper_url, json=request, timeout=self.request_timeout
                ) as response:
                    end_time = time.time()
                    if response.status == 200:
                        result = await response.json()
                        break
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                print("Request timed out")
            except Exception:
                action = "skipping" if i == self.n_retries - 1 else "ignoring"
                print(f"Error from _request ({action})")
                print(textwrap.indent(traceback.format_exc(), "  "))

        return chunk, result, end_time - start_time

    def _reduce_chunk(
        self, chunk: List[JSONType], result: Optional[JSONType], elapsed_time: float
    ):
        self._n_requests += 1

        if not result:
            self._n_failed += len(chunk)
            return

        # Validate
        assert len(result["outputs"]) == len(chunk)
        assert "billed_time" in result["profiling"]

        n_successful = sum(1 for r in result["outputs"] if r)
        self._n_successful += n_successful
        self._n_failed += len(chunk) - n_successful
        self._n_chunks_per_mapper[result["worker_id"]] += 1

        self._profiling["total_time"].push(elapsed_time)
        for k, vs in result["profiling"].items():
            for v in vs:
                self._profiling[k].push(v)

        if result["chunk_output"]:
            self.reducer.handle_chunk_result(chunk, result["chunk_output"])

        for input, output in zip(chunk, result["outputs"]):
            if output:
                self.reducer.handle_result(input, output)
