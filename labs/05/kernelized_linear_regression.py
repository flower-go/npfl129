#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=50, type=int, help="Number of examples")
    parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
    parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
    parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
    parser.add_argument("--iterations", default=5, type=int, help="Number of training iterations")
    parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    def compute_kernel(data1, data2):
        if args.kernel == 'rbf':
            return create_rbf(data1,data2)
        elif args.kernel == 'poly':
            return  create_poly(data1, data2)

    def create_rbf(data1, data2):
        #: K(x, y;gamma) = exp ^ {- gamma * | | x - y | | ^ 2}
        kernel = np.zeros((len(data1),len(data2)))
        for i in range(len(data1)):
            for j in range(len(data2)):
                kernel[i][j] = np.exp(-args.kernel_gamma * ((data1[i] - data2[j])**2))
        return kernel

    def create_poly(data1, data2):
    #poly: K(x, y; degree, gamma) = (gamma * x ^ T y + 1) ^ degree
        kernel = np.zeros((len(data1), len(data2)))
        for i in range(len(data1)):
            for j in range(len(data2)):
                kernel[i][j] = (args.kernel_gamma * data1[i].T*(data2[j]) + 1)**args.kernel_degree
        return kernel

    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.examples)
    train_targets = np.sin(5 * train_data) + np.random.normal(scale=0.25, size=args.examples) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.examples)
    test_targets = np.sin(5 * test_data)+ 1

    coefs = np.zeros(args.examples)

    kernel_train = compute_kernel(train_data, train_data)
    kernel_test = compute_kernel(train_data,test_data).T
    bias = np.mean(train_targets)

    # TODO: Perform `iterations` of SGD-like updates, but in dual formulation
    # using `coefs` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is MSE with L2 regularization:
    #   L = sum_{i=1}^N [1/2 * (target_i - phi(x_i)^T w - bias)^2] + 1/2 * args.l2 * w^2
    # Perform the update by optimizing this exact loss computed over
    # all training data (so it is a full GD algorithm, no batches or sampling).
    #
    # For bias use explicitly the average of training targets, and do not update
    # it futher during training.
    #
    # Instead of using feature map `phi` directly, we use given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following kernels:
    # - poly: K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - rbf: K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each update print RMSE (root mean squared error, i.e.,
    # sqrt(avg_i[(target_i-prediction_i)^2])) both on training and testing data.
    for iteration in range(args.iterations):
        # TODO
        #vygenerovat permutaci podle ktere budu upravovat
        permutation = np.random.permutation(len(coefs))
        cf = coefs.copy()

        for i in permutation:
            cf[i] = coefs[i] + args.learning_rate*(train_targets[i] - coefs.dot(kernel_train[i]) - bias) - args.learning_rate*(args.l2 * coefs[i])
        coefs = cf


        train_predictions = kernel_train.dot(coefs) + bias
        test_predictions =  kernel_test.dot(coefs)  + bias
        print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
            iteration + 1,
            np.sqrt(mean_squared_error(train_targets, train_predictions)),
           np.sqrt(mean_squared_error(test_targets, test_predictions)),
        ))

    if args.plot:
        test_predictions = ... #TODO

        plt.plot(train_data, train_targets, "bo", label="Train targets")
        plt.plot(test_data, test_targets, "ro", label="Test targets")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend(loc="upper left")
        plt.show()
