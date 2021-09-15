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

# Services:
# 1. Start db (or not if sqlite?)
# 1. Start django server
# 2. Start embedding server
# 3. Serve react static files -- sanic?

LOG_DIR = Path("~/.forager/logs").expanduser()
os.environ.setdefault("FORAGER_LOG_DIR", str(LOG_DIR))
os.environ.setdefault("FORAGER_LOG_STD", "1")
os.environ.setdefault("DEBUG_FRONTEND", "0")

DEBUG_FRONTEND = os.environ.get("DEBUG_FRONTEND") == "1"


def run_server(q):
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
    try:
        import django
        import uvicorn
        from django.core.management import call_command

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "forager_backend.settings")
        django.setup()

        if os.environ.get("FORAGER_LOG_STD"):
            logger = logging.getLogger("forager_backend")
            sys.stdout = LoggerWriter(logger.debug)
            sys.stderr = LoggerWriter(logger.warning)

        try:
            import forager_backend_api.migrations

            migrations_path = os.path.dirname(
                os.path.abspath(forager_backend_api.migrations.__file__)
            )
        except ModuleNotFoundError:
            import forager_backend_api

            migrations_path = os.path.join(
                os.path.dirname(os.path.abspath(forager_backend_api.__file__)),
                "migrations",
            )

        # Clear old migrations and recreate
        shutil.rmtree(migrations_path, ignore_errors=True)
        os.makedirs(migrations_path)
        with open(os.path.join(migrations_path, "__init__.py"), "w") as f:
            f.write(" ")

        call_command("makemigrations", "forager_backend_api")
        call_command("makemigrations")
        call_command("migrate")

        print("Running backend server...")
        q.put([True])
        uvicorn.run("forager_backend.asgi:application", host="0.0.0.0", port=8000)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


def run_embedding_server(q):
    try:
        from forager_embedding_server.log import init_logging

        init_logging()

        from forager_embedding_server import app

        print("Running embedding server...")
        q.put([True])
        app.app.run(host="0.0.0.0", port=5000)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


def run_frontend(q):
    try:
        from forager_frontend.log import init_logging

        init_logging()

        import uvicorn

        print("Running frontend...")
        q.put([True])
        uvicorn.run("forager_frontend.app:app", host="0.0.0.0", port=4000)
    except Exception:
        print(traceback.format_exc())
    finally:
        q.put([False])


class ForagerApp(object):
    web_server: Process
    embedding_server: Process
    file_server: Process

    web_server_q: Queue
    embedding_server_q: Queue
    file_server_q: Queue

    def __init__(self):
        pass

    def _run_server(self, fn) -> Tuple[Process, Queue]:
        q = Queue()
        p = Process(target=fn, args=(q,), daemon=True)
        p.start()
        return p, q

    def run(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print("Starting up Forager...")

        self.services: List[Tuple[str, Queue]] = []

        web_server_wd = ""
        self.web_server, self.web_server_q = self._run_server(run_server)
        self.services.append(("Backend", self.web_server_q))

        embedding_server_wd = ""
        self.embedding_server, self.embedding_server_q = self._run_server(
            run_embedding_server
        )
        self.services.append(("Compute", self.embedding_server_q))

        if not DEBUG_FRONTEND:
            file_server_wd = ""
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

        print("Forager is ready at: http://localhost:4000")
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
