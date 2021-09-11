import logging
import logging.config
import os
import sys
from io import IOBase


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


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "{asctime} - {name} - {levelname} - {message}",
            "style": "{",
        },
        "sanic": {
            "format": "%(asctime)s - (%(name)s)[%(levelname)s][%(host)s]: %(request)s %(message)s %(status)d %(byte)d",
            "style": "%",
        },
    },
    "handlers": {},
    "root": {"handlers": [], "level": "DEBUG"},
    "loggers": {
        "sanic.access": {"handlers": [], "propagate": False},
        "forager_frontend": {"handlers": [], "propagate": True},
    },
}

if os.environ.get("FORAGER_LOG_DIR"):
    path = os.environ["FORAGER_LOG_DIR"]
    LOGGING["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": os.path.join(path, "frontend_server.log"),
        "level": "DEBUG",
        "formatter": "simple",
    }
    LOGGING["root"]["handlers"].append("file")
    LOGGING["handlers"]["sanic_file"] = {
        "class": "logging.FileHandler",
        "filename": os.path.join(path, "frontend_server.log"),
        "level": "DEBUG",
        "formatter": "sanic",
    }
    LOGGING["loggers"]["sanic.access"]["handlers"].append("sanic_file")

if os.environ.get("FORAGER_LOG_CONSOLE") == "1":
    LOGGING["handers"]["console"] = {
        "class": "logging.StreamHandler",
        "level": "DEBUG",
        "formatter": "simple",
    }
    LOGGING["root"]["handlers"].append("console")
    LOGGING["handers"]["sanic_console"] = {
        "class": "logging.StreamHandler",
        "level": "DEBUG",
        "formatter": "sanic",
    }
    LOGGING["loggers"]["sanic.access"]["handlers"].append("sanic_console")


def init_logging():
    logging.config.dictConfig(LOGGING)

    if os.environ.get("FORAGER_LOG_STD") == "1":
        logger = logging.getLogger("forager_frontend")
        sys.stdout = LoggerWriter(logger.debug)
        sys.stderr = LoggerWriter(logger.warning)
