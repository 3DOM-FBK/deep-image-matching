class ConstantScheduler:
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value
