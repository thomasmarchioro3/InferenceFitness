import numpy as np

def evaluate_metrics(y_pred, y_true):
    # TODO: add AUC-ROC evaluation
    tp = np.sum(np.logical_and(y_pred == 0, y_true == 0) )
    fp = np.sum(np.logical_and(y_pred == 0, y_true == 1) )
    fn = np.sum(np.logical_and(y_pred == 1, y_true == 0) )
    tn = np.sum(np.logical_and(y_pred == 1, y_true == 1) )

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    balanced_accuracy = (recall+specificity)/2

    return accuracy, precision, recall, specificity, balanced_accuracy
