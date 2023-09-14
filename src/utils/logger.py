import json
import sys
import traceback
from logging import DEBUG, INFO, WARNING, Filter, Formatter, StreamHandler, getLogger


class JsonFormatter(Formatter):
    def format(self, record):
        return json.dumps(
            {
                "name": record.name,
                "level": record.levelname,
                "message": record.msg,
                "timestamp": self.formatTime(record, self.datefmt),
                "traceback": traceback.format_exc() if record.exc_info else [],
            },
            ensure_ascii=False,
        )


class HighPassFilter(Filter):
    def filter(self, record):
        return record.levelno >= int(WARNING)


class LowPassFilter(Filter):
    def filter(self, record):
        return record.levelno <= int(INFO)


def setup_logger(logger_name: str):
    formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    stdout_stream = StreamHandler(stream=sys.stdout)
    stdout_stream.setFormatter(formatter)
    stdout_stream.addFilter(LowPassFilter())

    stderr_stream = StreamHandler(stream=sys.stderr)
    stderr_stream.setFormatter(formatter)
    stderr_stream.addFilter(HighPassFilter())

    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)
    logger.addHandler(stdout_stream)
    logger.addHandler(stderr_stream)

    return logger
