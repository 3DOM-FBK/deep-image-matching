import logging
import time
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger("dim")


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
        logger.info(f"Function {func.__name__} took {total_time:.4f} s.")
        return result

    return timeit_wrapper


class Timer:
    """
    Class to help manage printing simple timing of code execution.

    Attributes:
        smoothing (float): A smoothing factor for the time measurements.
        cumulate_by_key (bool): Whether to accumulate time by key instead of smoothing.
        times (OrderedDict): A dictionary to store timing information for different named sections.
        will_print (OrderedDict): A dictionary to track whether to print the timing information for each section.
        logger (logging.Logger): The logger object to use for printing timing information.
    """

    def __init__(
        self,
        smoothing: float = 0.3,
        cumulate_by_key: bool = False,
        logger: logging.Logger | None = None,
    ):
        """
        Initializes the Timer object.

        Args:
            smoothing (float, optional): A smoothing factor for the time measurements. Defaults to 0.3.
            cumulate_by_key (bool, optional): Whether to accumulate time by key. Defaults to False.
            logger (logging.Logger, optional): The logger object to use for printing timing information. Defaults to None.
        """
        # Initial time values
        self.start_time = None
        self.last_time = None
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.smoothing = smoothing
        self.cumulate_by_key = cumulate_by_key
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.reset()

    def start(self):
        """
        Starts the Timer by resetting the initial time values.
        """
        self.reset()

    def reset(self):
        """
        Resets the Timer object, setting initial time values.
        """
        now = time.time()
        self.start_time = now
        self.last_time = now
        self.times.clear()
        self.will_print.clear()

    def update(self, name: str):
        """
        Updates the timing information for a specific named section. If the section does not exist, it is created,
        otherwise the timing information is updated. If cumulate_by_key was set to True, the timing information
        is accumulated for each key, otherwise the timing information is smoothed.

        Args:
            name (str): The name of the section.
        """
        # Guard against None to satisfy static type checkers and avoid runtime errors.
        now = time.time()
        dt = now - self.last_time if self.last_time is not None else 0.0
        self.last_time = now

        if name in self.times:
            if self.cumulate_by_key:
                self.times[name] += dt
            else:
                self.times[name] = (
                    self.smoothing * dt + (1 - self.smoothing) * self.times[name]
                )
        else:
            self.times[name] = dt

        self.will_print[name] = True

    def print(self, text: str = "Timer", sep: str = ", "):
        """
        Prints the accumulated timing information.

        Args:
            text (str, optional): Additional text to include in the printed message. Defaults to "Timer".
            sep (str, optional): Separator between timing entries. Defaults to ", ".
        """
        total = 0.0
        msg = f"[Timer] | [{text}] "
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                msg = msg + f"{key}={val:.3f}{sep}"
                total += val

        now = time.time()
        exec_time = now - self.start_time if self.start_time is not None else 0.0
        msg = msg + f"Total execution={exec_time:.3f}"
        self.logger.info(msg)

        self.reset()


if __name__ == "__main__":
    # Configure the logger
    logging.basicConfig(level=logging.INFO)

    # Test the Timer with multiple sections
    timer = Timer(smoothing=0.3)
    timer.start()

    time.sleep(0.1)
    timer.update("Section1")
    time.sleep(0.2)
    timer.update("Section2")
    time.sleep(0.3)
    timer.update("Section3")
    timer.print("Demo")

    # Test the Timer with cumulative mode
    timer_cumulative = Timer(cumulate_by_key=True)
    timer_cumulative.start()

    time.sleep(0.1)
    timer_cumulative.update("Task1")
    time.sleep(0.2)
    timer_cumulative.update("Task2")
    time.sleep(0.1)
    timer_cumulative.update("Task1")  # This should add to previous Task1
    timer_cumulative.print("Cumulative Demo")

    # Test the timeit decorator
    @timeit
    def example_function():
        time.sleep(0.5)

    example_function()
