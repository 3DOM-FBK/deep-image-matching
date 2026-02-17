from ripe import utils

log = utils.get_pylogger(__name__)


class LinearWithPlateaus:
    """Linear scheduler with plateaus.

    Linearly increases from `start_val` to `end_val`.
    Stays at `start_val` for `plateau_start_steps` steps and at `end_val` for `plateau_end_steps` steps.
    Linearly changes from `start_val` to `end_val` during the remaining steps.
    """

    def __init__(
        self,
        start_val,
        end_val,
        steps_total,
        rel_length_start_plateau=0.0,
        rel_length_end_plateu=0.0,
    ):
        self.start_val = start_val
        self.end_val = end_val
        self.steps_total = steps_total
        self.plateau_start_steps = steps_total * rel_length_start_plateau
        self.plateau_end_steps = steps_total * rel_length_end_plateu

        assert self.plateau_start_steps >= 0
        assert self.plateau_end_steps >= 0
        assert self.plateau_start_steps + self.plateau_end_steps <= self.steps_total

        self.slope = (end_val - start_val) / (steps_total - self.plateau_start_steps - self.plateau_end_steps)

        log.info(
            f"LinearWithPlateaus: start_val={start_val}, end_val={end_val}, steps_total={steps_total}, "
            f"plateau_start_steps={self.plateau_start_steps}, plateau_end_steps={self.plateau_end_steps}"
        )

    def __call__(self, step):
        if step < self.plateau_start_steps:
            return self.start_val
        if step < self.steps_total - self.plateau_end_steps:
            return self.start_val + self.slope * (step - self.plateau_start_steps)
        return self.end_val
