import pandas as pd
import os
import time

DATA_DIR = os.path.join(os.getcwd(), 'data')


def _csv_to_hdf5(csv_path, hdf5_path, sep=',', chunksize=1E6, force_delete=False):
    reader = pd.read_csv(csv_path, sep=sep, chunksize=chunksize)
    if os.path.exists(hdf5_path):
        if force_delete:
            print("Deleting existing HDF5 file %s" % hdf5_path)
            os.remove(hdf5_path)
        else:
            raise ValueError("File exists and force_delete is False. Exiting...")

    t0 = time.time()
    with pd.get_store(hdf5_path) as store:
        for n, chunk in enumerate(reader):
            try:
                nrows = store.get_storer('table').nrows
            except:
                nrows = 0
            chunk.index = pd.Series(chunk.index) + nrows
            store.append('table', chunk)
            print("Chunk", n, "time(s):", time.time() - t0)

    final = time.time() - t0
    print("Total time(s):", final)


def convert_train_csv_to_hdf5(force_delete=False):
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')
    train_hdf5_path = os.path.join(DATA_DIR, 'train.h5')
    _csv_to_hdf5(train_csv_path, train_hdf5_path, force_delete=force_delete)


def load_train_hdf5(train_hdf5_path=os.path.join(DATA_DIR, 'train.h5')):
    store = pd.HDFStore(train_hdf5_path, mode='r')
    return store


def convert_test_csv_to_hdf5(force_delete=False):
    test_csv_path = os.path.join(DATA_DIR, 'test.csv')
    test_hdf5_path = os.path.join(DATA_DIR, 'test.h5')
    _csv_to_hdf5(test_csv_path, test_hdf5_path, force_delete=force_delete)


def load_test_hdf5(test_hdf5_path=os.path.join(DATA_DIR, 'test.h5')):
    store = pd.HDFStore(test_hdf5_path, mode='r')
    return store


def retrieve_indices(store, start=None, stop=None, indices=None):
    if start is not None:
        assert stop is not None
        search_stringify = "index>=%i & index<%i" % (start, stop)
    elif indices is not None:
        assert start is None
        assert stop is None
        search_stringify = " or ".join(["index==%i" % i for i in indices])
    return store.select('table', where=[search_stringify])


if __name__ == "__main__":
    try:
        convert_train_csv_to_hdf5()
    except:
        print("Train file already exists!")
    try:
        convert_test_csv_to_hdf5()
    except:
        print("Test file already exists!")
