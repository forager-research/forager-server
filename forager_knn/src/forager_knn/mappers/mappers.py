import asyncio
from collections import defaultdict
import concurrent
from dataclasses import dataclass
import multiprocessing
import os

from typing import Any, DefaultDict, Dict, List, Optional, Type

from forager_knn import utils
from forager_knn.utils import JSONType

from .base import Mapper


@dataclass
class DummyRequest:
    json: JSONType


@utils.unasync
async def run_worker(
    cls: Type[Mapper],
    args: List[Any],
    kwargs: Dict[str, Any],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
):
    mapper = cls(*args, start_server=False, **kwargs)
    while True:
        request = input_queue.get()
        response = await mapper._handle_request(request)
        output_queue.put(response)


class MultiprocesingMapper(Mapper):
    def initialize_container(
        self, cls: Type[Mapper], nproc: int, *args, **kwargs
    ) -> None:
        self.nproc = nproc

        # First select even cores, then select odd cores if necessary
        cores = []
        next_core = 0
        available_cores = os.cpu_count()
        while nproc > 0:
            cores.append(next_core)
            nproc -= 1

            next_core += 2
            if available_cores and next_core >= available_cores:
                next_core = 1

        # Start workers in separate processes on CPU cores selected above
        self.input_queues: DefaultDict[int, multiprocessing.SimpleQueue] = defaultdict(
            multiprocessing.SimpleQueue
        )
        self.output_queues: DefaultDict[int, multiprocessing.SimpleQueue] = defaultdict(
            multiprocessing.SimpleQueue
        )

        for c in cores:
            proc = multiprocessing.Process(
                target=run_worker,
                args=(cls, args, kwargs, self.input_queues[c], self.output_queues[c]),
            )
            proc.start()
            os.system(f"taskset -p -c {c} {proc.pid}")

        self.worker_queue: Optional[asyncio.Queue[int]] = None

    def register_executor(self):
        return concurrent.futures.ThreadPoolExecutor(self.nproc)

    async def _handle_request(self, request):
        # Can't be done in initialize_container because requires async context
        # Doesn't need lock because never "await"s
        if self.worker_queue is None:
            self.worker_queue = asyncio.Queue()
            for worker_id in self.input_queues:
                self.worker_queue.put_nowait(worker_id)

        # Get an available worker
        worker_id = await self.worker_queue.get()

        # Send request to mapper (use DummyRequest to minimize data to be pickled)
        self.input_queues[worker_id].put(DummyRequest(request.json))

        # Wait for mapper to return a response
        response = await self.apply_in_executor(
            self.output_queues[worker_id].get,
            request_id="UNUSED",
            profiler_name="UNUSED",
        )

        # Mark the worker as available again
        self.worker_queue.put_nowait(worker_id)

        return response

    async def process_element(*args, **kwargs):
        raise NotImplementedError()
