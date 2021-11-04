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


DEFAULT_LEVEL = "DEBUG"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
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
    "root": {"handlers": [], "level": DEFAULT_LEVEL},
    "loggers": {
        "sanic.access": {"handlers": [], "propagate": False},
        "sanic.error": {"handlers": [], "propagate": True},
        "sanic.root": {"handlers": [], "propagate": True},
        "forager_embedding_server": {"handlers": [], "propagate": True},
    },
}

if os.environ.get("FORAGER_LOG_DIR"):
    path = os.environ["FORAGER_LOG_DIR"]
    LOGGING["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": os.path.join(path, "embedding_server.log"),
        "level": DEFAULT_LEVEL,
        "formatter": "simple",
    }
    LOGGING["root"]["handlers"].append("file")
    LOGGING["handlers"]["sanic_file"] = {
        "class": "logging.FileHandler",
        "filename": os.path.join(path, "embedding_server.log"),
        "level": DEFAULT_LEVEL,
        "formatter": "sanic",
    }
    LOGGING["loggers"]["sanic.access"]["handlers"].append("sanic_file")

if os.environ.get("FORAGER_LOG_CONSOLE") == "1":
    LOGGING["handlers"]["console"] = {
        "class": "logging.StreamHandler",
        "level": DEFAULT_LEVEL,
        "formatter": "simple",
    }
    LOGGING["root"]["handlers"].append("console")
    LOGGING["handlers"]["sanic_console"] = {
        "class": "logging.StreamHandler",
        "level": DEFAULT_LEVEL,
        "formatter": "sanic",
    }
    LOGGING["loggers"]["sanic.access"]["handlers"].append("sanic_console")

LOGGING_INIT = False


def init_logging():
    logging.config.dictConfig(LOGGING)
    if os.environ.get("FORAGER_LOG_STD") == "1":
        logger = logging.getLogger("forager_embedding_server")
        if os.environ.get("FORAGER_LOG_CONSOLE") == "1":
            logger.critical("Can't both log to console and log std out/err.")
            sys.exit(1)
        sys.stdout = LoggerWriter(logger.debug)
        sys.stderr = LoggerWriter(logger.warning)
