#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


class Node:
    def __init__(self, depth=None, indices=None, target=None):
        self.depth = depth
        self.indices = indices
        self.target = target
        self.feature = None
        self.feature_value = None
        self.left = None
        self.right = None

class Tree:
    def __init__(self, data, targets):
        self.root = Node(depth=0, indices=range(len(data)))
        self.data = data
        self.targets = targets

    def build(self):
        def __get_best_split(node):
            data = [self.data[i] for i in node.indices]
            max, l, r, feature, feature_value = None
            #pro kazdou feature
            for i in len(data[0]):
                #pro kazdy split point:
                data[data[:, 1].argsort()]
                for j in range(len(data[0,i]) - 1):
                    m = np.mean(data[j,i], data[j+1,i])
                    #compute metric
                    metric = ...
                    if max is None:
                        ...
                    elif max > metric:
                        ...

            ...

        def __split_by(split):
            feature, value, ind_a, ind_b = split
            ...

        def __split_more(node):
            ...

        def __get_node_prediction(data):
            u, c = np.unique(data, return_counts=True)
            return u[np.argmax(c)]

        nodes_for_split = [self.root]
        while(len(nodes_for_split) > 0):
            for_split = nodes_for_split.pop()
            #najdu nejlepsi split
            split = __get_best_split()
            #splitnu podle nej
            l,r = __split_by(split)
            #priradim
            for_split.left = l
            for_split.right = r
            #if depth je ok a velikost nodu je ok tak pridam do fronty
            if __split_more(l):
                nodes_for_split.append(l)
            else:
                l.target = __get_node_prediction(self.targets[l.indices])
            if __split_more(r):
                nodes_for_split.append(r)
            else:
                r.target = __get_node_prediction(self.targets[r.indices])








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
    parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
    parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
    parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=42, type=int, help="Test set size")
    args = parser.parse_args()


    def get_mean(targets):
        return np.mean(targets)


    def get_gini(data):
        unique, counts = np.unique(data, return_counts=True)
        return len(data) * (counts * (1 - counts))


    def get_entropy(data):
        unique, counts = np.unique(data)
        return -1 * len(data) * (counts * np.log(counts))


    def get_MSE(data):
        m = get_mean(data)
        return np.sum((data - m) ** 2)





    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a decision tree on the trainining data.
    tree=Tree(train_data, train_target).build()


    # - For each node, predict the most frequent class (and the one with
    # smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest feature values
    #   of the instances corresponding to the given node (i.e., for three instances
    #   with values 1, 7, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be at most `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).
    train_accuracy = 0
    test_accuracy = 0

    # TODO: Finally, measure the training and testing accuracy.
    print("Train acc: {:.1f}%".format(100 * train_accuracy))
    print("Test acc: {:.1f}%".format(100 * test_accuracy))
