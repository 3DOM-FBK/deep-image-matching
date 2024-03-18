# -------------------------------------------------------------------------------
#
#  Python dual-logging setup (console and log file),
#  supporting different log levels and colorized output
#
#  Created by Fonic <https://github.com/fonic>
#  Date: 04/05/20
#
#  Based on:                                                                   -
#  https://stackoverflow.com/a/13733863/1976617
#  https://uran198.github.io/en/python/2016/07/12/colorful-python-logging.html
#  https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
#
# -------------------------------------------------------------------------------

import functools
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path


def get_logger(name: str = None):
    logger = logging.getLogger(name)
    return logger


def change_logger_level(name: str = None, level: str = "warning"):
    logger = get_logger(name)
    log_level = logging.getLevelName(level.upper())
    for handler in logger.handlers:
        handler.setLevel(log_level)


def setup_logger(
    name: str = None,
    log_level: str = "info",
    log_folder: str = None,
    logfile_basename: str = "log",
) -> logging.Logger:
    """
    Configures and returns a logging.Logger instance.

    This function checks for existing loggers with the same name. It provides
    flexible configuration for both console and file-based logging with customizable
    log levels, formats, and an optional log file.

    Args:
        name (str, optional): The name of the logger. If None, the root logger
            will be used. Defaults to None.
        log_level (str, optional): The logging level for both console and file
            outputs. Valid options are 'debug', 'info', 'warning', 'error',
            'critical'. Defaults to 'info'.
        log_folder (str, optional): The path to the directory for storing log files.
            If None, no file output will be generated. Defaults to None.
        logfile_basename (str, optional): The base name for the log file. A timestamp
            will be appended. Defaults to "log".

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Check if logger already exists
    if logging.getLogger(name).hasHandlers():
        logger = logging.getLogger(name)
        logger.debug(f"Logger {logger.name} already exists")
        return logger

    # Set log line template
    if log_level == "debug":
        log_line_template = "%(color_on)s%(asctime)s | | [%(filename)s -> %(funcName)s], line %(lineno)d - [%(levelname)-8s] %(message)s%(color_off)s"
    else:
        log_line_template = (
            "%(color_on)s%(asctime)s | [%(levelname)-8s] %(message)s%(color_off)s"
        )

    # Set log file
    if log_folder is not None:
        log_folder = Path(log_folder)
        log_folder.mkdir(exist_ok=True, parents=True)
        today = date.today()
        now = datetime.now()
        current_date = f"{today.strftime('%Y_%m_%d')}_{now.strftime('%H:%M')}"
        log_file = log_folder / f"{logfile_basename}_{current_date}.log"
    else:
        log_file = None

    # Setup logging
    logger = configure_logging(
        name=name,
        console_log_output="stdout",
        console_log_level=log_level,
        console_log_color=True,
        logfile_file=log_file,
        logfile_log_level=log_level,
        logfile_log_color=False,
        log_line_template=log_line_template,
    )
    return logger


# Logging formatter supporting colorized output
class LogFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[1;33m",  # bright/bold yellow
        logging.INFO: "\033[0;37m",  # white / light gray
        logging.DEBUG: "\033[1;30m",  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if self.color is True and record.levelno in self.COLOR_CODES:
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)


# configure logging
def configure_logging(
    name,
    console_log_output,
    console_log_level,
    console_log_color,
    logfile_file,
    logfile_log_level,
    logfile_log_color,
    log_line_template,
) -> logging.Logger:
    # Create logger

    logger = logging.getLogger(name)

    # Set global log level to 'debug' (required for handler levels to work)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_log_output = console_log_output.lower()
    if console_log_output == "stdout":
        console_log_output = sys.stdout
    elif console_log_output == "stderr":
        console_log_output = sys.stderr
    else:
        print("Failed to set console output: invalid output: '%s'" % console_log_output)
        return False
    console_handler = logging.StreamHandler(console_log_output)

    # Set console log level
    try:
        console_handler.setLevel(
            console_log_level.upper()
        )  # only accepts uppercase level names
    except Exception as exception:
        print(
            f"Failed to set console log level: invalid level: {console_log_level}. {exception}"
        )
        return False

    # Create and set formatter, add console handler to logger
    datefmt = "%Y-%m-%d %H:%M:%S"
    console_formatter = LogFormatter(
        fmt=log_line_template, color=console_log_color, datefmt=datefmt
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create log file handler
    if logfile_file is not None:
        try:
            logfile_handler = logging.FileHandler(logfile_file)
        except Exception as exception:
            print(f"Failed to set up log file: {exception}")
            return False

        # Set log file log level
        try:
            logfile_handler.setLevel(
                logfile_log_level.upper()
            )  # only accepts uppercase level names
        except Exception as exception:
            print(
                f"Failed to set log file log level: invalid level: '{ logfile_log_level}. {exception}"
            )
            return False

        # Create and set formatter, add log file handler to logger
        logfile_formatter = LogFormatter(fmt=log_line_template, color=logfile_log_color)
        logfile_handler.setFormatter(logfile_formatter)
        logger.addHandler(logfile_handler)

    return logger


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    and a logging warning when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        message = kwargs.get("message", None)
        if message is None:
            message = f"Depracated {func.__name__}."
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        msg = f"Call to deprecated function {func.__name__}."
        warnings.warn(
            msg,
            category=DeprecationWarning,
            stacklevel=2,
        )
        logging.warning(msg)
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
