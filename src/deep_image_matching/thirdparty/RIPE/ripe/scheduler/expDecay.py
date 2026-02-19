import numpy as np

from ripe import utils

log = utils.get_pylogger(__name__)


class ExpDecay:
    """Exponential decay scheduler.
    args:
        a: float, a + c = initial value
        b: decay rate
        c: float, final value

        f(x) = a * e^(-b * x) + c
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        log.info(f"ExpDecay: a={a}, b={b}, c={c}")

    def __call__(self, step):
        return self.a * np.exp(-self.b * step) + self.c
