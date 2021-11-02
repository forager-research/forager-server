import asyncio
import os.path
from typing import List, Optional

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
def chunk_writer(url: str, file_path: str):

    chunk = yield
    with open(file_path, "wb") as f:
        while True:
            f.write(chunk)
            chunk = yield


async def download_file(
    session: aiohttp.ClientSession, url: str, sink, display_tqdm: Optional[bool] = False
):
    async with session.get(url) as response:
        if response.status != 200:
            raise FileNotFoundError(f"Url {url} returned response: {response}")
        assert response.status == 200
        if display_tqdm:
            prog = tqdm(
                total=response.content_length,
                unit="bytes",
                unit_scale=True,
                unit_divisor=1024,
            )
        try:
            while True:
                chunk = await response.content.read(CHUNK_SIZE)
                if not chunk:
                    break
                prog.update(len(chunk))
                sink.send(chunk)
        finally:
            if display_tqdm:
                prog.close()


async def as_completed_with_concurrency(n, tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    for x in asyncio.as_completed([sem_task(task) for task in tasks]):
        yield x


async def download_single(
    session: aiohttp.ClientSession,
    url: str,
    output_path: str,
    display_tqdm: Optional[bool] = False,
):

    await download_file(
        session,
        url,
        sink=chunk_writer(url, output_path),
        display_tqdm=display_tqdm,
    )


async def download_multiple(
    session: aiohttp.ClientSession,
    urls: List[str],
    output_directory: str,
    batch_size=16,
):
    download_futures = [
        download_file(
            session,
            url,
            sink=chunk_writer(
                url, os.path.join(output_directory, os.path.basename(url))
            ),
        )
        for url in urls
    ]
    async for download_future in tqdm(
        as_completed_with_concurrency(batch_size, download_futures),
        total=len(download_futures),
    ):
        result = await download_future
    return urls
