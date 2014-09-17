import gc
import os
import csv
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from utils import load_train_hdf5, load_test_hdf5, retrieve_indices

"""
uniq = {}
for c in minibatch.columns[2:]:
    try:
        if uniq[c].dtype == np.object:
            uniq[c] = pd.Series(np.concatenate((
                uniq[c], minibatch[c].fillna("NaN").unique()))).unique()
        else:
            uniq[c] = pd.Series(np.concatenate((
                uniq[c], minibatch[c].fillna(-1).unique()))).unique()
    except KeyError:
        uniq[c] = minibatch[c].unique()
"""


def get_row_count(store):
    # Dirty hack to get nrows
    return int([k.split("->")[-1] for k in
                str(store.get_storer('table')).split(",") if "nrows" in k][0])

store = load_train_hdf5()
minibatch = retrieve_indices(store, start=0, stop=1)
use_cols = [k for k in minibatch.columns[2:] if 'C' not in k]
t0 = time.time()
minibatch_size = 1E6
# There must be some better way to get the total number of rows...
train_row_count = get_row_count(store)
pts = np.arange(train_row_count, step=minibatch_size)
if pts[-1] != train_row_count:
    pts = np.append(pts, train_row_count)

dummy_predictor = []
for n, (i, j) in enumerate(zip(pts[:-1], pts[1:])):
    if (n % 5) == 0:
        print("Iteration ", n, ": time", time.time() - t0)
    minibatch = retrieve_indices(store, start=i, stop=j)
    X = minibatch[use_cols].fillna(-1).values
    y = minibatch['Label'].values
    dummy_predictor.append(y.mean())
    del X
    del y
    del minibatch
    gc.collect()

print("Total training time for all minibatches: ", time.time() - t0)

store = load_test_hdf5()
t0 = time.time()
minibatch_size = 1E6
test_row_count = get_row_count(store)
pts = np.arange(test_row_count, step=minibatch_size)
if pts[-1] != test_row_count:
    pts = np.append(pts, test_row_count)
results_name = 'submission.csv'
f = open(results_name, 'w')
f.write("Id,Predicted\n")
f.close()
f = open(results_name, 'ab')
for n, (i, j) in enumerate(zip(pts[:-1], pts[1:])):
    if (n % 5) == 0:
        print("Iteration ", n, ": time", time.time() - t0)
    minibatch = retrieve_indices(store, start=i, stop=j)
    X = minibatch[use_cols].fillna(-1).values
    pred_y = dummy_predictor[n] * np.ones_like(minibatch['Id'].values)
    results = np.vstack((minibatch['Id'].values.astype('int32'), pred_y)).T
    np.savetxt(f, results, fmt='%i,%0.6f')
    del X
    del pred_y
    del results
    del minibatch
    gc.collect()
f.close()
print("Total prediction time for all minibatches: ", time.time() - t0)
