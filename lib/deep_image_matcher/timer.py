import logging
import time

from functools import wraps
from collections import OrderedDict


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


class AverageTimer:
    """Class to help manage printing simple timing of code execution."""

    def __init__(self, smoothing=0.3, logger=None):
        self.smoothing = smoothing
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.logger = logger
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name="default"):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text="Timer"):
        total = 0.0
        msg = f"[Timer] | [{text}] "
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                msg = msg + f"%s=%.3f, " % (key, val)
                total += val
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

        self.reset()
