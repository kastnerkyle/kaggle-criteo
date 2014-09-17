import pandas
import numpy as np
import tables

I_labels = ['I%d' % i for i in range(1, 14)]
C_labels = ['C%d' % i for i in range(1, 27)]


train_df = pandas.read_table('train.csv', sep=',')
test_df = pandas.read_table('test.csv', sep=',')

len_train = len(train_df)
len_test = len(test_df)

# train_categorical_memmap = np.memmap('train_categorical.dat',
#                                      mode='w+',
#                                      shape=(len_train, len(C_labels)),
#     dtype=np.int32)
# test_categorical_memmap = np.memmap('train_categorical.dat',
#                                     mode='w+',
#                                     shape=(len_test, len(C_labels)),
#     dtype=np.int32)

f = tables.open_file('data.hf5', 'w')
atom = tables.Atom.from_dtype(np.dtype('int32'))
train_categorical = f.create_array(f.root, 'categorical_train', atom=atom,
                                   shape=(len_train, len(C_labels)))
test_categorical = f.create_array(f.root, 'categorical_test', atom=atom,
                                   shape=(len_test, len(C_labels)))


for i, C_label in enumerate(C_labels):
    print C_label
    train_test = np.concatenate([train_df[C_label].values,
                                 test_df[C_label].values])
    print "Concatenated train/test."
    unique_labels, indices = np.unique(train_test, return_inverse=True)
    fname = "feature_%s.npz" % C_label
    np.savez(fname, labels=unique_labels,
            train=indices[:len_train].astype(np.int32),
            test=indices[len_train:].astype(np.int32))
    print fname
    chunk_size = 1000000
    print "train data to memmap"
    # for j in xrange(0, len_train, chunk_size):
    #     print j
    #     train_categorical_memmap[:, i][j:j + chunk_size] = \
    #         indices[:len_train][j:j + chunk_size].astype(np.int32)
    train_categorical[:, i] = indices[:len_train]
    print "test data to memmap"
    # test_categorical_memmap[:, i] = indices[len_train:].astype(np.int32)
    test_categorical[:, i] = indices[len_train:]

f.close()


