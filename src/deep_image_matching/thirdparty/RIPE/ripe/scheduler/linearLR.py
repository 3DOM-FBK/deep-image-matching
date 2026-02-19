class StepLinearLR:
    """Decay the learning rate by a linearly changing factor at each STEP (not epoch).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (int): Total number of steps in the training process.
        initial_lr (float): Initial learning rate.
        final_lr (float): Final learning rate.
    """

    def __init__(self, optimizer, steps_init, num_steps, initial_lr, final_lr):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.i_step = steps_init
        self.decay_factor = (final_lr - initial_lr) / num_steps

    def step(self):
        """Decay the learning rate by decay_factor."""
        self.i_step += 1

        if self.i_step > self.num_steps:
            return

        lr = self.initial_lr + self.i_step * self.decay_factor
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_step(self):
        return self.i_step
