class BaseTrain(object):
    def __init__(self,config,model,train_datagen,val_datagen):
        self.model = model
        self.train_datagen = train_datagen
        self.val_datagen = val_datagen
        self.config = config

    def train(self):
        raise NotImplementedError
