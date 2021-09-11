import asyncio
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import shutil
import time
import uuid

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    DefaultDict,
    Set,
    TypeVar,
    Optional,
    MutableMapping,
)

import requests


KT = TypeVar("KT")
VT = TypeVar("VT")


class CleanupDict(MutableMapping[KT, VT]):
    def __init__(
        self,
        cleanup_func: Callable[[VT], Awaitable[None]],
        schedule_func: Optional[Callable[[Awaitable[Any]], Any]] = None,
        timeout: Optional[float] = None,
        interval: float = 60,
    ) -> None:
        self.cleanup_func = cleanup_func
        self.schedule_func = schedule_func
        self.timeout = timeout
        self.interval = interval

        self.store: Dict[KT, VT] = {}
        self.last_accessed: Dict[KT, float] = {}
        self.locks: DefaultDict[KT, Set[str]] = defaultdict(set)

        if self.schedule_func and self.timeout:
            self.schedule_func(self.sleep_and_cleanup())

    def __getitem__(self, key: KT) -> VT:
        value = self.store[key]
        self.last_accessed[key] = time.time()
        return value

    def __setitem__(self, key: KT, value: VT) -> None:
        self.store[key] = value
        self.last_accessed[key] = time.time()

    def __delitem__(self, key: KT) -> None:
        del self.store[key]
        del self.last_accessed[key]
        self.locks.pop(key, None)

    def __iter__(self):
        # Optimistically mark everything as accessed now
        for key in self.last_accessed:
            self.last_accessed[key] = time.time()
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def lock(self, key: KT, lock_name: Optional[str] = None) -> Optional[str]:
        if key not in self.store:
            return None
        lock_name = lock_name or str(uuid.uuid4())
        self.locks[key].add(lock_name)
        return lock_name

    def unlock(self, key: KT, lock_name: str) -> None:
        self.last_accessed[key] = time.time()
        if lock_name in self.locks[key]:
            self.locks[key].remove(lock_name)

    def clear(self) -> None:
        asyncio.run(self.clear_async())

    async def cleanup_key(self, key: KT) -> None:
        await self.cleanup_func(self.store.pop(key))
        del self.last_accessed[key]

    async def clear_async(self) -> None:
        self.locks.clear()
        await asyncio.gather(*map(self.cleanup_key, self.store.keys()))

    async def sleep_and_cleanup(self) -> None:
        assert self.schedule_func and self.timeout

        await asyncio.sleep(self.interval)

        current = time.time()
        keys_to_delete = [
            k
            for k, v in self.last_accessed.items()
            if current - v > self.timeout and not self.locks[k]
        ]
        await asyncio.gather(*map(self.cleanup_key, keys_to_delete))

        self.schedule_func(self.sleep_and_cleanup())


def sha_encode(x: str):
    hash_object = hashlib.sha256(x.encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return str(hex_dig)


def make_identifier(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_remote_file(url):
    filename = os.path.basename(url)
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r, path.open("wb") as f:
            shutil.copyfileobj(r.raw, f)
    return filename
