#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO(linear_regression_l2): Split the dataset randomly to train
    # and test using `sklearn.model_selection.train_test_split`, with
    # `test_size=args.test_ratio` and `random_state=args.seed`.
    train, test, train_y, test_y = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_ratio, random_state=args.seed)

    # TODO: Process the input columns in the following way:
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise).
    column_number = train.shape[1]
    categorical = []
    for i in range(column_number):
        is_int = all(x % 1 == 0 for x in dataset.data[:,i])
        if is_int:
            categorical.append(i)
    numerical = l3 = [x for x in range(column_number) if x not in categorical]
    #   Encode the values using one-hot encoding,
    #   probably using `sklearn.preprocessing.OneHotEncoder` (note that its output
    #   is by default sparse; you can use `sparse=False` to generate dense output)
    #   * * * * * Please ignore the warning OneHotEncoder * * * * *
    #   * * * * * prints and do not pass `categories=auto`* * * * *
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; you can use `sklearn.preprocessing.StandardScaler`.
    numeric_transformer = sklearn.pipeline.Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler(copy=False))])

    categorical_transformer = sklearn.pipeline.Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    # TODO: Generate polynomial features of order 2 from the current features.
    # If the input values are [a, b, c, d], you should generate
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocess.PolynomialFeatures(2, include_bias=False)`.
    poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical),
            ('num', numeric_transformer, numerical)
        ])

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.
    clf = sklearn.pipeline.Pipeline(steps=[('preprocessor', preprocessor),('poly', poly)])

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.
    train_data = clf.fit_transform(train)
    test_data = clf.transform(test)
    #train_data = train
    #test_data = test
    with open("feature_engineering.out", "w") as output_file:
        for data in [train_data, test_data]:
            for line in range(5):
                print(" ".join("{:.6g}".format(data[line, column]) for column in range(data.shape[1])), file=output_file)
