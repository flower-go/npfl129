#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--examples", default=200, type=int, help="Number of examples")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.5, type=float, help="Test set size ratio")
    args = parser.parse_args()


    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
    def sigmoid_array(a):
        return [sigmoid(x) for x in a.dot(weights)]
    def predict(d):
        return [1 if p >= 0.5 else 0 for p in d]
    def neg_log_likelihood(predictions, labels):
        # Take the error when label=1
        class1_cost = -labels * np.log(predictions)

        # Take the error when label=0
        class2_cost = (1 - np.array(labels)) * np.log(1 - np.array(predictions))

        # Take the sum of both costs
        cost = class1_cost - class2_cost

        # Take the average cost
        cost = cost.sum() / len(labels)

        return cost


    # Set random seed
    np.random.seed(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.examples, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_ratio` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_ratio, random_state=args.seed)

    # Generate initial linear regression weights
    weights = np.random.uniform(size=train_data.shape[1])

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        batches = [permutation[i:i + args.batch_size] for i in range(0, len(permutation), args.batch_size)]
        for indices in batches:
            x = train_data[indices]
            t = train_target[indices]
            gradients = []

            for i in range(len(t)):
                diff = (sigmoid(x[i].transpose().dot(weights)) - t[i])
                gradients.append(diff * x[i])

            weights = weights - args.learning_rate * np.mean(gradients, axis=0)

        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        # Print the accuracies in percentages.
        train_sig = sigmoid_array(train_data)
        test_sig = sigmoid_array(test_data)

        prediction_train = predict(train_sig)
        prediction_test = predict(test_sig)

        train_loss = neg_log_likelihood(train_sig, train_target)
        test_loss = neg_log_likelihood(test_sig, test_target)
        test_accuracy = accuracy_score(test_target, prediction_test)
        train_accuracy = accuracy_score(train_target, prediction_train)

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1,
            train_loss,
            100 * train_accuracy,
            test_loss,
            100 * test_accuracy,
        ))

        if args.plot:
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 20)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 20)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=40, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            plt.show()
