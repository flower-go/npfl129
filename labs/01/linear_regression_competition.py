#!/usr/bin/env python3
import argparse
import pickle

import sklearn as sklearn
import sklearn.linear_model
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="linear_regression_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the data to train["data"] and train["target"]
    train = np.load("linear_regression_competition.train.npz")
    train = {entry: train[entry] for entry in train}

    #for i in range(12):
    #   train['data'] = np.concatenate((train['data'],train['data'][:,i].reshape(len(train['data']),1)**2), axis = 1)
    #print(train['data'].shape)

    # TODO: Train the model
    model = RandomForestRegressor(max_depth=10, random_state=args.seed,n_estimators=100000)
    #model = model.fit(train['data'], train['target'])

    #model = sklearn.linear_model.Ridge(alpha=1)
        #.fit(train['data'], train['target'])
    print(np.sqrt(-1*cross_val_score(model, train['data'], train['target'], cv=10, scoring='neg_mean_squared_error')))

    # TODO: The trained model needs to be saved. All sklear models can
    # be serialized and deserialized using the standard `pickle` module.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with open(args.model_path, "wb") as model_file:
          pickle.dump(model, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a Numpy array containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with open('linear_regression_competition.model', "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    return np.array(model.predict(data))
