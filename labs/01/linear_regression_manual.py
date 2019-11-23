#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", default=50, type=int, help="Test size to use")
    args = parser.parse_args()

    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()
    print(dataset.DESCR)

    # The input data are in dataset.data, targets are in dataset.target.

    # TODO: Pad a value of `1` to every instance in dataset.data
    # (np.pad or np.concatenate might be useful).
    dataset.data = np.concatenate((dataset.data, np.ones(len(dataset.target)).reshape(len(dataset.target), 1)), axis=1)

    # TODO: Split data so that the last `args.test_size` data are the test
    # set and the rest is the training set.
    train, test = np.split(dataset.data, [len(dataset.target) - args.test_size])
    train_target, test_target = np.split(dataset.target, [len(dataset.target) - args.test_size])

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using np.linalg.inv).
    w = (inv(train.transpose().dot(train)).dot(train.transpose())).dot(train_target)

    # TODO: Predict target values on the test set.
    prediction = test.dot(w)

    # TODO: Compute root mean square error on the test set predictions.
    rmse = math.sqrt(mean_squared_error(test_target, prediction))

    with open("linear_regression_manual.out", "w") as output_file:
       print("{:.2f}".format(rmse), file=output_file)
