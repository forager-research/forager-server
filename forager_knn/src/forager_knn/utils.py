import asyncio
import base64
import concurrent
from dataclasses import dataclass, field
import functools
import io
import textwrap
import traceback

import aiohttp

from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Set,
    Union,
)

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


class FileListIterator:
    def __init__(self, list_path: str, map_fn=lambda line: line) -> None:
        self.map_fn = map_fn

        self._list = open(list_path, "r")
        self._total = 0
        for line in self._list:
            if not line.strip():
                break
            self._total += 1
        self._list.seek(0)

    def close(self, *args, **kwargs):
        self._list.close()

    def __len__(self):
        return self._total

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        elem = self._list.readline().strip()
        if not elem:
            raise StopIteration
        return self.map_fn(elem)


@dataclass
class _LimitedAsCompletedState:
    pending: List[asyncio.Future] = field(default_factory=list)
    hit_stop_iteration = False
    next_coro_is_pending = False


async def limited_as_completed_from_async_coro_gen(
    coros: AsyncIterable[Coroutine[Any, Any, Any]], limit: int
) -> AsyncGenerator[asyncio.Task, None]:
    state = _LimitedAsCompletedState()
    NEXT_CORO_TASK_NAME = "get_next_coro"

    async def get_next_coro():
        try:
            coro = await coros.__anext__()
            return coro
        except StopAsyncIteration:
            state.hit_stop_iteration = True

    def schedule_getting_next_coro():
        task = asyncio.create_task(get_next_coro(), name=NEXT_CORO_TASK_NAME)
        state.pending.append(task)
        state.next_coro_is_pending = True

    schedule_getting_next_coro()

    while state.pending:
        done_set, pending_set = await asyncio.wait(
            state.pending, return_when=asyncio.FIRST_COMPLETED
        )
        state.pending = list(pending_set)

        for done in done_set:
            assert isinstance(done, asyncio.Task)
            if done.get_name() == NEXT_CORO_TASK_NAME:
                state.next_coro_is_pending = False
                if state.hit_stop_iteration:
                    continue

                # Schedule the new coroutine
                state.pending.append(asyncio.create_task(done.result()))

                # If we have capacity, also ask for the next coroutine
                if len(state.pending) < limit:
                    schedule_getting_next_coro()
            else:
                # We definitely have capacity now, so ask for the next coroutine if
                # we haven't already
                if not state.next_coro_is_pending and not state.hit_stop_iteration:
                    schedule_getting_next_coro()

                yield done


# https://stackoverflow.com/a/50029150
async def as_completed_from_futures(
    tasks: Iterable[asyncio.Future],
) -> AsyncGenerator[asyncio.Future, None]:
    pending_set: Set[asyncio.Future] = set(tasks)
    while pending_set:
        done_set, pending_set = await asyncio.wait(
            pending_set, return_when=asyncio.FIRST_COMPLETED
        )
        for done in done_set:
            yield done


async def run_in_executor(f, *args, executor: concurrent.futures.Executor = None):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, f, *args)


def unasync(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


def unasync_as_task(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        # Surface exceptions when they occur, not when awaited (makes debugging easier)
        new_coro = log_exception_from_coro_but_return_none(coro)
        return asyncio.create_task(new_coro(*args, **kwargs))

    return wrapper


def log_exception_from_coro_but_return_none(coro):
    @functools.wraps(coro)
    async def wrapper(*args, **kwargs):
        try:
            return await coro(*args, **kwargs)
        except asyncio.CancelledError:
            pass
        except Exception:
            print(f"Error from {coro.__name__}")
            print(textwrap.indent(traceback.format_exc(), "  "))
        return None

    return wrapper


def numpy_to_base64(nda):
    import numpy as np

    if nda is None:
        return ""

    with io.BytesIO() as nda_buffer:
        np.save(nda_buffer, nda, allow_pickle=False)
        nda_bytes = nda_buffer.getvalue()
        nda_base64 = base64.b64encode(nda_bytes).decode("ascii")
    return nda_base64


def base64_to_numpy(nda_base64):
    import numpy as np

    if not nda_base64:
        return None

    nda_bytes = base64.b64decode(nda_base64)
    with io.BytesIO(nda_bytes) as nda_buffer:
        nda = np.load(nda_buffer, allow_pickle=False)
    return nda


def create_unlimited_aiohttp_session() -> aiohttp.ClientSession:
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    return aiohttp.ClientSession(connector=conn)
