import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from base.base_evaluation import BaseEvaluation
from sklearn.metrics import classification_report



class EvaluationBinary(BaseEvaluation):
    def __init__(self, config,x_test,y_test,model):
        super(EvaluationBinary, self).__init__(config,x_test,y_test,model)
        self.model = model
        self.x_test = x_test
        self.y_test = y_test 
        
        self.predict_x=self.model.predict(self.x_test) 
        self.y_pred=np.argmax(self.predict_x,axis=1)
        print("Clasification report")
        print(classification_report(self.y_test, self.y_pred, digits = 4))
        
        #y_test= test_datagen.classes

        #y_true, y_pred
            
    def plot_confusion_matrix(self, classes =np.r_[0,1],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("confusion matrix:",cm)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(self.y_test, self.y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig("confusion_matrix.png")
        return ax
        