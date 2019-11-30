#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

def softmax(z):
    p = np.exp(z)
    suma = np.sum(p)
    return p / suma

def relu_der(x):
    return [1 if i > 0 else 0 for i in x]

def relu_der_matrix(m):
    result = np.zeros_like(m)
    for i in range(m.shape[0]):
            result[i,:] = relu_der(m[i])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
    parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Hidden layer size")
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
    weights = [np.random.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               np.random.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.

        hidden = inputs.dot(weights[0])
        hidden = np.maximum(hidden,0)

        output = hidden.dot(weights[1])
        output = softmax(output - np.max(output))

        return output,hidden
        #
        # The value of the hidden layer is computed as ReLU(inputs times weights[0]).
        # The value of the output layer is computed as softmax(hidden_layer times weights[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

    for iteration in range(args.iterations):
        permutation = np.random.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now two parts, gradient of weghts[0] and weights[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it "step by step" using chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer (we did this already in softmax_classification_sgd)
        # - compute the derivative with respect to weights[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0]
        batches = [permutation[i:i + args.batch_size] for i in range(0, len(permutation), args.batch_size)]
        for indices in batches:
            x = train_data[indices]
            t = train_target[indices]
            update1 = np.zeros_like(weights[1])
            update0 = np.zeros_like(weights[0])
            for i in range(len(t)):
                o,h = forward(x[i])
                dL_sinput = (o - [1 if j == t[i] else 0 for j in range(10)])
                dLoss_dh_out = weights[1].dot(dL_sinput)
                dH_out_H_in = h > 0
                dH_in_wij = x[i].reshape([-1, 1])

                update1 += np.outer(h,dL_sinput)
                update0 += dLoss_dh_out*dH_out_H_in*dH_in_wij

            update1 = update1/len(t)
            update0 = update0/len(t)
            weights[1] -= args.learning_rate * update1
            weights[0] -= args.learning_rate * update0

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_acc = 0
        test_acc = 0

        train_pred = [forward(inp)[0] for inp in train_data]
        train_pred = [np.argmax(i) for i in train_pred]
        test_pred = [forward(inp)[0] for inp in test_data]
        test_pred = [np.argmax(i) for i in test_pred]

        train_acc = sklearn.metrics.accuracy_score(train_target, train_pred)
        test_acc = sklearn.metrics.accuracy_score(test_target, test_pred)
        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1,
            100 * train_acc,
            100 * test_acc,
        ))
