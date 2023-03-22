class BaseEvaluation(object):
    def __init__(self, config):
        self.config = config

    def get_confusion_matrix(self):
        raise NotImplementedError