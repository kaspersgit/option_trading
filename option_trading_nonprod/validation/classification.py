import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
import itertools

def plotCurveAUC(probs, actual, title, type='roc', savefig=False, saveFileName='test.png'):
    """
    :param probs: float models predicted probability
    :param actual: int 1 or 0 for the actual outcome
    :param title: str title of the plot
    :param type: str either ROC or PR
    :param savefig: boolean if plot should be saved
    :param saveFileName: str where to save the plot to
    :return:
    """
    type = type.lower()
    if type == 'roc':
        xVar, yVar, thresholds = metrics.roc_curve(actual, probs)
        title = 'ROC curve - {}'.format(title)
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate (Recall)'
        diagCor1 = [0,1]
        diagCor2 = [0,1]
    elif ('recall' in type and 'precision' in type) or type == 'pr':
        yVar, xVar, thresholds = metrics.precision_recall_curve(actual, probs)
        title = 'Precision-Recall curve - {}'.format(title)
        xlabel = 'True Positive Rate (Recall)'
        ylabel = 'Precision'
        diagCor1 = [1,0]
        diagCor2 = [0,1]
        
    # Calculate area under curve (AUC)
    auc = metrics.auc(xVar, yVar)

    plt.figure()
    lw = 2
    plt.plot(xVar, yVar, color='darkgreen',
            lw=lw, label=title + ' (area = %0.2f)' % auc)
    plt.plot(diagCor1, diagCor2, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    if savefig:
        plt.savefig(saveFileName)
    return(auc)

def plotThresholdMetrics(pred, actual, savefig=False, saveFileName='pr-threshold.png'):
    precision, recall, th = metrics.precision_recall_curve(actual, pred)
    plt.plot(th, precision[1:], label="Precision", linewidth=5)
    plt.plot(th, recall[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()
    if savefig:
        plt.savefig(saveFileName)


def showConfusionMatrix(pred, actual, normalize=None):
    """
    pred: the predicted classes
    actual: the actual class
    normalize: normalize matrix ['true', 'pred', None]    
    """
    # confusion matrix 
    matrix = metrics.confusion_matrix(actual,pred, normalize=normalize)
    class_names = ['incorrect','correct']
    plt.clf()

    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')

    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # plot colorbar to the right
    plt.colorbar()

    if normalize == None:
        fmt = 'd'
    else:
        fmt = 'f'

    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        i = int(i)
        j = int(j)
        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label',size=14)
    plt.xlabel('Predicted label',size=14)
    plt.show()
