#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import pandas as pd
import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

class Dataset:
    def __init__(self,
                 name="binary_classification_competition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `train_target` (column Target)
        # and `train_data` (all other columns).
        dataset = pd.read_csv(name)
        self.data, self.target = dataset.drop("Target", axis=1), dataset["Target"]


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="binary_classification_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the dataset, downloading it if required
    train = Dataset()

    # Note that `train.data` and `train.target` are a `pandas.DataFrame`.
    # It is similar to a Numpy array, but columns have names and types.
    # You can get column types using `train_data.dtypes`; note that
    # strings are actually encoded as `object`s.

    # TODO: Train the model.
    numeric_transformer = sklearn.pipeline.Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler(copy=False))])

    categorical_transformer = sklearn.pipeline.Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

    numerical = list(train.data.select_dtypes([np.number]).columns)
    categorical = [x for x in train.data.columns if x not in numerical]



    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical),
            ('num', numeric_transformer, numerical)
        ])
    clf = sklearn.pipeline.Pipeline(steps=[('preprocessor', preprocessor), ('poly', poly),('pred', LinearRegression())])
    train_data = clf.fit(train.data, train.target)

    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we can also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
          pickle.dump(clf, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a pandas.DataFrame containt test set input.

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # TODO: Return the predictions as a Numpy array.
    prediction = model.predict(data)
    prediction = [1 if x > 0.5 else 0 for x in prediction]
    return np.array(prediction)

print(mean_squared_error(train.data,recodex_predict(train.data)))
