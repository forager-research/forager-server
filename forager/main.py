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


def run_cmd():
    from forager.app import ForagerApp

    app = ForagerApp()
    app.run()
    app.join()


def dev_cmd():
    pass


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")

    dev_parser = subparsers.add_parser("dev", help="development commands for forager")

    dev_parser.set_defaults(func=dev_cmd)

    args = parser.parse_args()
    if args.subparser_name is None:
        run_cmd()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
