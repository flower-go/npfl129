#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    print(dataset.DESCR, file=sys.stderr)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_ratio, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    min_max = sklearn.pipeline.Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', MinMaxScaler())])
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    poly = sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(multi_class="multinomial", random_state=args.seed)
    logistic = sklearn.linear_model.LogisticRegression(multi_class="multinomial", random_state=args.seed)

    pipe = sklearn.pipeline.Pipeline(steps=[('min_max',min_max), ('poly',poly), ('logistic', logistic)])
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    param_grid = {
        'poly__degree': [1,2],
        'logistic__C':[0.01,1,100],
        'logistic__solver': ['lbfgs', 'sag']
    }

    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    search.fit(train_data, train_target)
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.
    best = search.best_estimator_

    test_accuracy = accuracy_score(test_target, best.predict(test_data))
    print("{:.2f}".format(100 * test_accuracy))
