import pandas as pd
import tables
import time
import numpy as np

def get_row_count(fpath):
    with open(fpath) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

t0 = time.time()
row_count = get_row_count('data/train.csv')
print(row_count)
print(time.time() - t0)
f = tables.open_file('labels.h5', 'w')
atom = tables.Atom.from_dtype(np.dtype('float32'))
train_label = f.create_array(f.root, 'label', atom=atom,
                             shape=(row_count, 1))
from IPython import embed; embed()
