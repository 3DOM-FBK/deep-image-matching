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

import logging
import sys
from datetime import date, datetime
from pathlib import Path
import functools
import warnings


def setup_logger(
    console_log_level: str = "info",
    log_folder: str = None,
    logfile_level: str = "info",
    logfile_basename: str = "log",
) -> logging.Logger:
    if log_folder is not None:
        log_folder = Path(log_folder)
        log_folder.mkdir(exist_ok=True, parents=True)
        today = date.today()
        now = datetime.now()
        current_date = f"{today.strftime('%Y_%m_%d')}_{now.strftime('%H:%M')}"
        log_file = log_folder / f"{logfile_basename}_{current_date}.log"
    else:
        log_file = None

    # log_line_template = "%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s"

    if console_log_level == "debug" or logfile_level == "debug":
        log_line_template = "%(color_on)s%(asctime)s | | [%(filename)s -> %(funcName)s], line %(lineno)d - [%(levelname)-8s] %(message)s%(color_off)s"
    else:
        log_line_template = (
            "%(color_on)s%(asctime)s | [%(levelname)-8s] %(message)s%(color_off)s"
        )

    # Setup logging
    if not configure_logging(
        console_log_output="stdout",
        console_log_level=console_log_level,
        console_log_color=True,
        logfile_file=log_file,
        logfile_log_level=logfile_level,
        logfile_log_color=False,
        log_line_template=log_line_template,
    ):
        print("Failed to setup logging, aborting.")
        raise RuntimeError

    return get_logger()


def get_logger(name: str = "__name__"):
    logger = logging.getLogger(name)
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
    console_log_output,
    console_log_level,
    console_log_color,
    logfile_file,
    logfile_log_level,
    logfile_log_color,
    log_line_template,
):
    # Create logger
    # For simplicity, we use the root logger, i.e. call 'logging.getLogger()'
    # without name argument. This way we can simply use module methods for
    # for logging throughout the script. An alternative would be exporting
    # the logger, i.e. 'global logger; logger = logging.getLogger("<name>")'
    logger = logging.getLogger()

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
    except:
        print(
            "Failed to set console log level: invalid level: '%s'" % console_log_level
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
            print("Failed to set up log file: %s" % str(exception))
            return False

        # Set log file log level
        try:
            logfile_handler.setLevel(
                logfile_log_level.upper()
            )  # only accepts uppercase level names
        except:
            print(
                "Failed to set log file log level: invalid level: '%s'"
                % logfile_log_level
            )
            return False

        # Create and set formatter, add log file handler to logger
        logfile_formatter = LogFormatter(fmt=log_line_template, color=logfile_log_color)
        logfile_handler.setFormatter(logfile_formatter)
        logger.addHandler(logfile_handler)

    # Success
    return True


# Call main function
if __name__ == "__main__":
    CONSOLE_LOG_LEVEL = "info"
    LOGFILE_LEVEL = "info"
    LOG_FOLDER = "logs"
    LOG_BASE_NAME = "icepy4d"

    # Setup logger
    setup_logger(LOG_FOLDER, LOG_BASE_NAME, CONSOLE_LOG_LEVEL, LOGFILE_LEVEL)

    # Log some messages
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical message")
