from scipy.io import loadmat
import numpy as np

def load_label(file_name='labels.mat',var_name='label'):
    annots = loadmat(file_name)[var_name]
    labels = []
    for k in annots:
        labels.append(k[0])
    labels = np.array(labels)
    return labels

def load_data(file_name='dataset.mat',var_name='data'):
    annots = loadmat(file_name)[var_name]
    annots = annots[0]
    data = list()

    for traj in annots:
        row = []
        for k in traj:
            row.append(k[0])

        data.append(row)
    data = np.array(data)
    return data





