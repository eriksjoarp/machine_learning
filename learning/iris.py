#!/usr/bin/env python3


#########################################
#
#   Goal
#   Learn simple ML techniques with the iris dataset
#
#
#
#
#
#
#
#
#
#
#
#
#########################################

import sys, os, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

path = os.path.join(os.getcwd(), os.pardir, 'my_modules')
sys.path.append(os.path.abspath(path))

import helper as h
import ml_helper as ml_h
import ml_helper_visualization as ml_viz
import constants as c

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


#############################################
PROJECT = 'mnist'

DATASET_DIR = os.path.join(c.DATASET_DIR, PROJECT)


#############################################

START_TIME = time.time()


mnist = fetch_openml('mnist_784', version=1, data_home=DATASET_DIR)

print(mnist.keys())

X, y = mnist['data'], mnist['target']

print(X.shape)
print(y.shape)
print(type(mnist))

print(y[0])
y = y.astype(np.int8)

#some_digit_image = some_digit.reshape[28,28]

#plt.imshow(some_digit_image, cmap='binary')
#plt.axis('off')
#plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print(str(time.time() - START_TIME))

sgd_clf = SGDClassifier(shuffle=True, random_state=c.RANDOM_STATE)
sgd_clf.fit(X_train, y_train_5)

skfolds = StratifiedKFold(n_splits=3, shuffle= True, random_state=c.RANDOM_STATE)

print(str(time.time() - START_TIME))

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[train_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct /len(y_pred))

print(str(time.time() - START_TIME))
