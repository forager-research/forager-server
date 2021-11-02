import asyncio
import logging
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from io import IOBase
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Tuple

import readchar

import forager.demo_dataset_paths as demo_dataset_paths
from forager.buildutils import FRONTEND_DIR
from forager.utils import create_unlimited_aiohttp_session, download_single

# Services:
# 1. Start db (or not if sqlite?)
# 1. Start django server
# 2. Start embedding server
# 3. Serve react static files -- sanic?

FORAGER_DIR = Path("~/.forager/").expanduser()
NOT_FIRST_RUN_FILE = Path("~/.forager/not_first_run.dummy").expanduser()
LOG_DIR = Path("~/.forager/logs").expanduser()
DEMO_DATASET_DIR = Path("~/.forager/demo_dataset").expanduser()
os.environ.setdefault("FORAGER_LOG_DIR", str(LOG_DIR))
os.environ.setdefault("FORAGER_LOG_STD", "1")
os.environ.setdefault("DEBUG_FRONTEND", "0")

DEBUG_FRONTEND = os.environ.get("DEBUG_FRONTEND") == "1"
FRONTEND_PORT = 4000
BACKEND_PORT = 8000
EMBEDDING_SERVER_PORT = 5000


def run_server(q, dev):
    class LoggerWriter(IOBase):
        def __init__(self, writer):
            self._writer = writer
            self._msg = ""

        def write(self, message):
            self._msg = self._msg + message
            while "\n" in self._msg:
                pos = self._msg.find("\n")
                self._writer(self._msg[:pos])
                self._msg = self._msg[pos + 1 :]

        def flush(self):
            if self._msg != "":
                self._writer(self._msg)
                self._msg = ""

    # Run migrations if needed
    if dev:
        os.environ["FORAGER_DEV"] = "1"
        os.environ["FORAGER_LOG_CONSOLE"] = "1"
        os.environ["FORAGER_LOG_STD"] = "0"
    try:
        import django
        import uvicorn
        from django.core.management import call_command

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "forager_backend.settings")
        django.setup()

        if os.environ.get("FORAGER_LOG_STD") == "1":
            logger = logging.getLogger("forager_backend")
            sys.stdout = LoggerWriter(logger.debug)
            sys.stderr = LoggerWriter(logger.warning)

        import forager_backend_api

        migrations_path = os.path.join(
            os.path.dirname(os.path.abspath(forager_backend_api.__file__)),
            "migrations",
        )
        if not os.path.exists(migrations_path):
            os.makedirs(migrations_path)
            with open(os.path.join(migrations_path, "__init__.py"), "w") as f:
                f.write(" ")

        call_command("makemigrations", "forager_backend_api")
        call_command("makemigrations")
        call_command("migrate")

        print("Running backend server...")
        q.put([True])
        uvicorn.run(
            "forager_backend.asgi:application", host="0.0.0.0", port=BACKEND_PORT
        )
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


def run_embedding_server(q, dev):
    if dev:
        os.environ["FORAGER_DEV"] = "1"
        os.environ["FORAGER_LOG_CONSOLE"] = "1"
        os.environ["FORAGER_LOG_STD"] = "0"
    try:
        from forager_embedding_server.log import init_logging

        init_logging()

        from forager_embedding_server import app

        print("Running embedding server...")
        q.put([True])
        app.app.run(host="0.0.0.0", port=EMBEDDING_SERVER_PORT)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


def run_frontend(q, dev):
    assert not dev
    try:
        from forager_frontend.log import init_logging

        init_logging()

        import uvicorn

        print("Running frontend...")
        q.put([True])
        uvicorn.run("forager_frontend.app:app", host="0.0.0.0", port=FRONTEND_PORT)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


def dev_frontend(q, dev):
    class TempSymlink:
        def __init__(self, src, dest):
            self.src = src
            self.dest = dest

        def __enter__(self):
            if os.path.exists(self.dest) or os.path.islink(self.dest):
                if os.path.islink:
                    os.unlink(self.dest)
                else:
                    raise FileExistsError(
                        f"Attempted to symlink {self.dest} to {self.src}, but {self.dest} exists and is not a symlink."
                    )
            os.symlink(self.src, self.dest)

        def __exit__(self, type, value, traceback):
            os.unlink(self.dest)

    assert dev
    os.environ.setdefault("FORAGER_DEV", "1")
    try:
        import forager_frontend
        from forager_frontend.log import init_logging

        init_logging()

        os.environ.setdefault(
            "REACT_APP_SERVER_URL", f"http://localhost:{BACKEND_PORT}"
        )
        os.environ.setdefault("PORT", f"{FRONTEND_PORT}")
        print("Running frontend...")
        q.put([True])
        cwd = os.path.join(
            os.path.dirname(os.path.realpath(forager_frontend.__file__)), "../node"
        )
        subprocess.run("npm install", cwd=cwd, shell=True)
        with TempSymlink("/", os.path.join(cwd, "public/files")):
            subprocess.run("npm run start", cwd=cwd, shell=True)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


class ForagerApp(object):
    dev: bool
    run_backend: bool
    run_frontend: bool
    is_first_run: bool

    web_server: Process
    embedding_server: Process
    file_server: Process

    web_server_q: Queue
    embedding_server_q: Queue
    file_server_q: Queue

    def __init__(
        self, run_backend: bool = True, run_frontend: bool = True, dev: bool = False
    ):
        self.run_backend = run_backend
        self.run_frontend = run_frontend
        self.dev = dev
        if not os.path.exists(NOT_FIRST_RUN_FILE):
            self.is_first_run = True
        else:
            self.is_first_run = False

    def _run_server(self, fn, dev: bool = False) -> Tuple[Process, Queue]:
        q = Queue()
        p = Process(target=fn, args=(q, dev), daemon=True)
        p.start()
        return p, q

    async def _add_initial_dataset(self):
        # Download images
        dataset_name = "demo_dataset"
        url = demo_dataset_paths.get_archive_path()
        output_directory = str(DEMO_DATASET_DIR)
        http_session = create_unlimited_aiohttp_session()
        print("Downloading demo dataset...")
        os.makedirs(output_directory, exist_ok=True)
        await download_single(
            http_session,
            url,
            os.path.join(output_directory, os.path.basename(url)),
            display_tqdm=True,
        )
        # Extract dataset
        proc = await asyncio.create_subprocess_exec(
            "tar",
            "-xzf",
            os.path.join(output_directory, os.path.basename(url)),
            "-C",
            output_directory,
        )
        await proc.wait()
        # Add dataset
        request = dict(
            dataset=dataset_name,
            train_images_directory=os.path.join(output_directory, "train_images"),
            val_images_directory=os.path.join(output_directory, "val_images"),
        )
        print("Adding demo dataset to Forager...")
        async with http_session.post(
            f"http://localhost:{BACKEND_PORT}/api/create_dataset",
            json=request,
            timeout=1200,
        ) as response:
            if not response.ok:
                print("Error creating dataset: ", response.reason)
                response.raise_for_status()
            j = await response.json()
            if j["status"] != "success":
                print("Error adding embeddings: ", j["reason"])
                raise RuntimeError(j["reason"])
        # Import embeddings
        for name in ["resnet", "clip"]:
            embeddings_dir = os.path.join(output_directory, "model_outputs", name)
            image_list_path = os.path.join(embeddings_dir, "image_list.txt")
            paths = []
            with open(image_list_path, "r") as f:
                # Remap the paths
                paths = [
                    os.path.join(output_directory, x.split(" ")[2].strip())
                    for x in f.readlines()
                ]
            embeddings_path = os.path.join(embeddings_dir, "embeddings.bin")
            params = {
                "name": name,
                "paths": paths,
                "embeddings_path": str(embeddings_path),
            }
            async with http_session.post(
                f"http://localhost:{BACKEND_PORT}/api/add_model_output/{dataset_name}",
                json=params,
            ) as response:
                if not response.ok:
                    print("Error creating dataset: ", response.reason)
                    response.raise_for_status()
                j = await response.json()
                if j["status"] != "success":
                    print("Error adding embeddings: ", j["reason"])
                    raise RuntimeError(j["reason"])

        print("Success! Dataset added as 'demo_dataset'.")

    def run(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print("Starting up Forager...")

        self.services: List[Tuple[str, Queue]] = []

        if self.run_backend:
            web_server_wd = ""
            self.web_server, self.web_server_q = self._run_server(
                run_server, dev=self.dev
            )
            self.services.append(("Backend", self.web_server_q))

            embedding_server_wd = ""
            self.embedding_server, self.embedding_server_q = self._run_server(
                run_embedding_server, self.dev
            )
            self.services.append(("Compute", self.embedding_server_q))

        if self.run_frontend:
            file_server_wd = ""
            if self.dev:
                self.file_server, self.file_server_q = self._run_server(
                    dev_frontend, dev=True
                )
                self.services.append(("Frontend", self.file_server_q))
            else:
                self.file_server, self.file_server_q = self._run_server(run_frontend)
                self.services.append(("Frontend", self.file_server_q))

        for idx, (name, q) in enumerate(self.services):
            started = q.get()
            if started:
                print(f"({idx+1}/{len(self.services)}) {name} started.")
                pass
            if not started:
                print(f"{name} failed to start, aborting...")
                sys.exit(1)

        if (
            self.is_first_run
            and self.run_backend
            and not (self.dev and self.run_frontend)
        ):
            if self.dev:
                # Wait for 3 seconds for sanic output to resolve
                time.sleep(3)
            print(
                "This is your first time running Forager. Would you like to download a test dataset? (Y/n) ",
                end=" ",
                flush=True,
            )
            while True:
                inp = readchar.readchar()
                print(inp)
                if inp == "Y" or inp == "y":
                    yes = True
                    break
                elif inp == "N" or inp == "n":
                    yes = False
                    break
                elif inp == "\x03":
                    raise KeyboardInterrupt()
                print('\nPlease specify "y" or "n". ', end=" ", flush=True)
            if yes:
                asyncio.run(self._add_initial_dataset())

            with open(str(NOT_FIRST_RUN_FILE), "w") as f:
                f.write(" ")

        print(f"Forager is ready at: http://localhost:{FRONTEND_PORT}")
        print(f"(Logs are in {LOG_DIR})")

    def join(self):
        aborting = False
        self.service_running = [True for _ in self.services]
        self.services_left = len(self.services)
        while self.services_left > 0:
            for idx, (name, q) in enumerate(self.services):
                if not self.service_running[idx]:
                    continue
                try:
                    still_running = q.get_nowait()
                    if not still_running:
                        self.service_running[idx] = False
                        self.services_left -= 1
                        if not aborting:
                            print(f"{name} failed, aborting...")
                            aborting = True
                except queue.Empty:
                    pass
            time.sleep(1)

        self.web_server.join()
        self.embedding_server.join()
        self.file_server.join()
