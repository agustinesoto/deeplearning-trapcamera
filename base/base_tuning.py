class BaseTuning(object):
    def __init__(self, config):
        self.config = config

    def get_models(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
