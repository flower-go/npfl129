#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import argparse
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class Dataset:
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        dataset = np.load(name)
        self.data, self.target = dataset["data"].reshape([-1, 28 * 28]).astype(np.float), dataset["target"]


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # TODO: Train the model.
    # logistic = LogisticRegression(multi_class='multinomial', random_state=args.seed)
    # param_grid = {
    #     'logistic__C': [0.01, 1, 100],
    #     'logistic__solver': ['lbfgs', 'sag']
    # }
    # pipe = sklearn.pipeline.Pipeline(steps=[('logistic', logistic)])
    # search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    # search.fit(train.data, train.target)
    # model = search.best_estimator_
    # model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=args.seed).fit(train.data, train.target)

    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=50/len(train.target))
    # scores = cross_val_score(model, train.data, train.target, cv=5)
    # print(scores)
    # model = model.fit(train.data, train.target)

    steps = [('scaler', StandardScaler()), ('svm', SVC(kernel='poly', C=0.1, gamma=1))]
    pipeline = sklearn.pipeline.Pipeline(steps)

    # parameters = {'svm__C': [0.001, 0.1, 100, 10e5], 'svm__gamma': [10, 1, 0.1, 0.01]}
    #     # grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    #     #
    #     # grid.fit(train.data, train.target)
    #     #
    #     # model = grid.best_estimator_
    #     # print(grid.best_score_)
    pipeline.fit(train.data, train.target)

    model = pipeline

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)


# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a numpy arrap containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    return np.array(model.predict(data))
