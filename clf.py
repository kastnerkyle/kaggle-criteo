"""
Modified from code by: https://www.kaggle.com/users/185835/tinrtgu
Forum post: https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory
"""

# Authors: tinrtgu
#          Kyle Kastner
# License: BSD 3 Clause

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt


# parameters #################################################################

train = 'train.csv'  # path to training file
rev_train = 'rev_train.csv'
test = 'test.csv'  # path to testing file

D = 2 ** 20   # number of weights use for learning
alpha = .1    # learning rate for sgd optimization


# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 1e-6), 1e-6)
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Label': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    x = [0]  # 0 is the index of the bias term
    for key, value in csv_row.items():
        index = int(value + key[1:], 16) % D  # weakest hash ever ;)
        x.append(index)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


def training_loop(w, n, reverse=False):
    loss = 0.
    if not reverse:
        dr = DictReader(open(train))
    else:
        dr = DictReader(open(rev_train))

    for t, row in enumerate(dr):
        y = 1. if row['Label'] == '1' else 0.

        del row['Label']  # can't let the model peek the answer
        del row['Id']  # we don't need the Id

        # main training procedure
        # step 1, get the hashed features
        x = get_x(row, D)

        # step 2, get prediction
        p = get_p(x, w)

        # for progress validation, useless for learning our model
        loss += logloss(p, y)
        if t % 1000000 == 0 and t > 1:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), t, loss/t))

        # step 3, update model with answer
        w, n = update_w(w, n, x, p, y)
    print("Total number of lines %d" % t)
    return w, n

# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

for i in range(2):
    w, n = training_loop(w, n, reverse=True)

# testing (build kaggle's submission file)
with open('submission.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test))):
        Id = row['Id']
        del row['Id']
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%s,%f\n' % (Id, p))
