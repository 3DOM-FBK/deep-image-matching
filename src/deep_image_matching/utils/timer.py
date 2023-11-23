import logging
import time
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger("deep-image-matching")


def timeit(func):
    """
    Decorator that measures the execution time of a function and prints the duration.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} s.")
        return result

    return timeit_wrapper


class Timer:
    """
    Class to help manage printing simple timing of code execution.

    Attributes:
        smoothing (float): A smoothing factor for the time measurements.
        times (OrderedDict): A dictionary to store timing information for different named sections.
        will_print (OrderedDict): A dictionary to track whether to print the timing information for each section.
        logger (logging.Logger): The logger object to use for printing timing information.
    """

    def __init__(
        self,
        smoothing: float = 0.3,
        logger: logging.Logger = logger,
        log_level: str = "info",
    ):
        """
        Initializes the Timer object.

        Args:
            smoothing (float, optional): A smoothing factor for the time measurements. Defaults to 0.3.
            logger (logging.Logger, optional): The logger object to use for printing timing information. Defaults to logger.
        """
        self.smoothing = smoothing
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.logger = logger
        self.log_level = logging.getLevelName(log_level.upper())

        self.reset()

    def reset(self):
        """
        Resets the Timer object, setting initial time values.
        """
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name: str):
        """
        Updates the timing information for a specific named section.

        Args:
            name (str): The name of the section.
        """
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text: str = "Timer", sep: str = ", "):
        """
        Prints the accumulated timing information.

        Args:
            text (str, optional): Additional text to include in the printed message. Defaults to "Timer".
        """
        total = 0.0
        msg = f"[Timer] | [{text}] "
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                msg = msg + f"{key}={val:.3f}{sep}"
                total += val
        msg = msg + f"Total={total:.3f}"
        # self.logger.info(msg)
        self.logger.log(self.log_level, msg)

        self.reset()


if __name__ == "__main__":
    # Configure the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("deep-image-matching")

    # Function to test with the timeit decorator
    @timeit
    def example_function():
        time.sleep(1)

    # Timer instance with custom logger and log level
    custom_logger = logging.getLogger("custom-logger")
    timer = Timer(smoothing=0.3, logger=custom_logger, log_level="info")

    # Test the Timer with multiple sections
    time.sleep(0.1)
    timer.update("Section1")
    time.sleep(0.2)
    timer.update("Section2")
    time.sleep(0.3)
    timer.update("Section3")
    timer.print("Demo")

    # Test the Timer with the timeit decorator
    example_function()
