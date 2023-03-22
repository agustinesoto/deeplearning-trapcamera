class BaseEvaluation(object):
    def __init__(self, config,x_test,y_test,model):
        self.config = config
        self.x_test = x_test
        self.y_test = y_test
        self.model = model


    def plot_confusion_matrix(self):
        raise NotImplementedError
    

