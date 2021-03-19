import pandas as pd
import numpy as np
import os

from preprocess_pmdata import preprocess_pmdata as preprocess, format_ts_data
from model.LSTMClassifier import LSTMClassifier
from utils import evaluate_metrics

datapath = os.path.join("data", "pmdata")

if __name__ == '__main__':
    timesteps = 7
    strides = 2

    def rule(info):
        return 1*np.array(info['age'] > 30 )

    x_data, y_data = preprocess(datapath, rule)

    # IMP: if you use shuffle=True the users from training will be the same from test
    # which may lead to wrong results (i.e., the model learns users' behavior)
    x_data, y_data = format_ts_data(x_data, y_data, timesteps, strides, shuffle=False)

    Tsplit = int(0.5*len(x_data)) # use 0.5 of the data for training
    x_train = x_data[:Tsplit]
    y_train = y_data[:Tsplit]
    x_test = x_data[Tsplit:]
    y_test = y_data[Tsplit:]

    model = LSTMClassifier(n_classes=2, timesteps=timesteps, n_features=2)

    # TODO: implement training until convergence
    model.fit(x_train, y_train, epochs=200)

    # compute metrics on training set
    y_pred_train = model.predict(x_train)
    accuracy, precision, recall, specificity, balanced_accuracy = evaluate_metrics(y_pred_train, y_train)

    print('')
    print('RESULTS ON TRAINING SET')
    print("Accuracy train: {0:.2f}".format(accuracy))
    print("Precision train: {0:.2f}".format(precision))
    print("Recall train: {0:.2f}".format(recall))
    print("Specificity train: {0:.2f}".format(specificity))
    print("Balanced accuracy train: {0:.2f}".format(balanced_accuracy))
    print('')

    # compute metrics on test set
    y_pred_test = model.predict(x_test)
    accuracy, precision, recall, specificity, balanced_accuracy = evaluate_metrics(y_pred_test, y_test)

    print('RESULTS ON TEST SET')
    print("Accuracy test: {0:.2f}".format(accuracy))
    print("Precision test: {0:.2f}".format(precision))
    print("Recall test: {0:.2f}".format(recall))
    print("Specificity test: {0:.2f}".format(specificity))
    print("Balanced accuracy test: {0:.2f}".format(balanced_accuracy))
    print('')
