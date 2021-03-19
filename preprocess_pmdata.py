import pandas as pd
import numpy as np
import os

def preprocess_pmdata(datapath, label_rule, attributes=['steps', 'calories'], p=0.5, q=-1):
    x = []
    subfolders = [dd[1] for dd in os.walk(datapath)][0]

    for subdir in subfolders:
        file = os.path.join(datapath, subdir, "daily.csv")
        df = pd.read_csv(file)
        x.append(df[attributes].values)

    info = pd.read_csv(os.path.join(datapath, "participants_info.csv"), delimiter='\t')
    y = label_rule(info)

    return x, y

def format_ts_data(x_data, y_data, timesteps=7, strides=2, shuffle=True, normalize_std=True):
    x = []
    y = []
    for xx, yy in zip(x_data, y_data):
        x_temp = np.array([xx[i:i+timesteps] for i in range(0, len(xx)-timesteps, strides)])
        y_temp = yy*np.ones(len(x_temp))
        x.append(x_temp)
        y.append(y_temp)

    x = np.concatenate(x)
    y = np.concatenate(y)

    if normalize_std:
        x = x/np.std(x)

    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]

    return x, y
