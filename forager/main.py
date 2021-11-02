import argparse
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


def run_cmd(args):
    from forager.app import ForagerApp

    app = ForagerApp()
    app.run()
    app.join()


def dev_cmd(args):
    from forager.app import ForagerApp

    run_frontend = False
    run_backend = False
    if args.frontend:
        run_frontend = True
    if args.backend:
        run_backend = True
    if not args.frontend and not args.backend:
        run_frontend = True
        run_backend = True

    app = ForagerApp(run_frontend=run_frontend, run_backend=run_backend, dev=True)
    app.run()
    app.join()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")

    dev_parser = subparsers.add_parser("dev", help="development commands for forager")
    dev_parser = subparsers.add_parser("dev", help="development commands for forager")
    dev_parser.add_argument("--frontend", action="store_true")
    dev_parser.add_argument("--backend", action="store_true")

    dev_parser.set_defaults(func=dev_cmd)

    args = parser.parse_args()
    if args.subparser_name is None:
        run_cmd(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
