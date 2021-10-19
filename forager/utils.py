import asyncio
import os.path
from typing import List

import aiohttp
from tqdm.asyncio import tqdm

CHUNK_SIZE = 4 * 1024  # 4 KB


def create_unlimited_aiohttp_session() -> aiohttp.ClientSession:
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    return aiohttp.ClientSession(connector=conn)


def coroutine(func):
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr

    return start


@coroutine
def chunk_writer(url: str, output_directory: str):
    file_path = os.path.join(output_directory, os.path.basename(url))

    chunk = yield
    with open(file_path, "wb") as f:
        while True:
            f.write(chunk)
            chunk = yield


async def download_file(session: aiohttp.ClientSession, url: str, sink):
    async with session.get(url) as response:
        assert response.status == 200
        while True:
            chunk = await response.content.read(CHUNK_SIZE)
            if not chunk:
                break
            sink.send(chunk)
    return url


async def as_completed_with_concurrency(n, tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    for x in asyncio.as_completed([sem_task(task) for task in tasks]):
        yield x


async def download_multiple(
    session: aiohttp.ClientSession,
    urls: List[str],
    output_directory: str,
    batch_size=16,
):
    download_futures = [
        download_file(session, url, sink=chunk_writer(url, output_directory))
        for url in urls
    ]
    async for download_future in tqdm(
        as_completed_with_concurrency(batch_size, download_futures),
        total=len(download_futures),
    ):
        result = await download_future
    return urls
