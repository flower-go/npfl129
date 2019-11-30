#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import accuracy_score

def softmax(z, label):
    w = weights[:,label]
    p = np.exp(z.transpose().dot(w))
    suma = np.sum([np.exp(z.transpose().dot(weights[:,i])) for i in range(len(weights[0]))])
    return p/suma

def gradient(z, label):
    num_classes = len(weights[0])
    scores = np.zeros(num_classes)
    for l in range(num_classes):
        w = weights[:,l]
        scores[l] = w.dot(z)
    suma = np.sum(np.exp(scores))

    percent_exp_score = np.exp(scores) / suma
    for j in range(num_classes):
        grad[:, j] += percent_exp_score[j] * z
    grad[:, label] -= z

    return grad

def softmax_array(z):
    result = []
    for x in z:
        row = []
        for i in range(len(weights[0])):
            row.append(softmax(x, i))
        result.append(row)
    return result


def prediction(z):
    z = np.array(z)
    return z.argmax(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=797, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = np.random.uniform(size=[train_data.shape[1], args.classes])

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        batches = [permutation[i:i + args.batch_size] for i in range(0, len(permutation), args.batch_size)]
        for indices in batches:
            x = train_data[indices]
            t = train_target[indices]
            grad = np.zeros_like(weights)

            for i in range(len(t)):
                gradient(x[i], t[i])

            grad /= len(t)
            weights -= args.learning_rate * grad

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_soft = softmax_array(train_data)
        test_soft = softmax_array(test_data)

        prediction_train = prediction(train_soft)
        prediction_test = prediction(test_soft)

        train_accuracy = 0
        test_accuracy = 0

        test_accuracy = accuracy_score(test_target, prediction_test)
        train_accuracy = accuracy_score(train_target, prediction_train)


        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * train_accuracy,
            100 * test_accuracy,
        ))
