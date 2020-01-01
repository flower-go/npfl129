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
        self.criterion = None

class Tree:
    def __init__(self, data, targets):
        self.root = Node(depth=0, indices=range(len(data)))
        self.data = data
        self.targets = targets
        self.root.criterion = Tree.get_gini(self.targets[self.root.indices])
        self.leaves = 0

    def get_gini(data):
        unique, counts = np.unique(data, return_counts=True)
        no_data = len(data)
        return no_data * np.sum((counts / no_data) * (1 - (counts / no_data)))

    def get_entropy(data):
        unique, counts = np.unique(data, return_counts=True)
        return -1 * len(data) * np.sum((counts/len(data)) * np.log((counts/len(data))))

    def build(self):

        def get_criterion(targets, features,value, feature):
            if args.criterion == "gini":
                #TODO nepouzivat indexy
                return (Tree.get_gini(targets[features[:,feature] <= value]), Tree.get_gini(targets[features[:,feature] > value]))
            else:
                return (Tree.get_entropy(targets[features[:,feature] <= value]), Tree.get_entropy(targets[features[:,feature] > value]))


        def __get_best_split(node):
            data = np.array([self.data[i] for i in node.indices])
            targets = self.targets[node.indices]
            max, l, r, feature, feature_value = None, None, None, None, None
            #pro kazdou feature
            for i in range(len(data[0])):
                #pro kazdy split point:
                i_sorted = data[:, i].argsort()
                sorted = data[i_sorted]
                sorted_targets = targets[i_sorted]
                for j in range(len(sorted) - 1):
                    m = np.mean([sorted[j,i], sorted[j+1,i]])
                    #compute metric
                    metric_l, metric_r = get_criterion(sorted_targets, sorted, m, i)
                    if max is None:
                        l = metric_l
                        r = metric_r
                        max = l + r
                        feature = i
                        feature_value = m
                    elif max > metric_l + metric_r:
                        l = metric_l
                        r = metric_r
                        max = l + r
                        feature = i
                        feature_value = m
            return feature, feature_value, max, l, r

        def __split_by(split, node):
            feature, value, criterion, l_c, r_c = split
            l = [index for index in node.indices if self.data[index,feature] <= value]
            r = [index for index in node.indices if self.data[index,feature] > value]
            node.feature = feature
            node.feature_value = value
            l  = Node(depth=node.depth + 1, indices = l)
            r = Node(depth=node.depth + 1, indices= r)
            l.criterion = l_c
            r.criterion = r_c

            return l,r

        def __split_more(node):
            if args.max_depth and node.depth >= args.max_depth:
                return False
            if args.min_to_split and len(node.indices) < args.min_to_split:
                return False
            if node.criterion == 0:
                return False
            else:
                return True


        def __get_node_prediction(data):
            u, c = np.unique(data, return_counts=True)
            return u[np.argmax(c)]

        def __split_given_node(node):
            # najdu nejlepsi split
            split = __get_best_split(for_split)
            # splitnu podle nej
            l, r = __split_by(split, for_split)

            # if for_split.feature is None:
            #     for_split.target = __get_node_prediction(self.targets[for_split.indices])
            #     continue
            # priradim
            for_split.left = l
            for_split.right = r
            # if depth je ok a velikost nodu je ok tak pridam do zasobniku
            if __split_more(r):
                nodes_for_split.append(r)
            else:
                r.target = __get_node_prediction(self.targets[r.indices])
                self.leaves += 1
            if __split_more(l):
                nodes_for_split.append(l)
            else:
                l.target = __get_node_prediction(self.targets[l.indices])
                self.leaves += 1

        nodes_for_split = [self.root]
        while(len(nodes_for_split) > 0):
            if  args.max_leaves and self.leaves + 2 + len(nodes_for_split) - 1 > args.max_leaves:
                for rest in nodes_for_split:
                    rest.target = __get_node_prediction(self.targets[rest.indices])
                    self.leaves += 1
                break
            if args.max_leaves is None or len(nodes_for_split) == 1:
                for_split = nodes_for_split.pop()
                __split_given_node(for_split)
            else:
                min_c = None
                min_i = None
                split_i = None
                for i in range(len(nodes_for_split)):
                    node = nodes_for_split[i]
                    split = __get_best_split(node)
                    feature, value, criterion, l_c, r_c = split
                    if min_c is None or min_c > (l_c + r_c) - node.criterion:
                        min_c =  l_c + r_c - node.criterion
                        min_i = i
                        split_i = split
                for_split  = nodes_for_split.pop(min_i)
                __split_given_node(for_split)






    def predict(self, data):
        predictions = []
        for i in data:
            n = self.root
            while n.target is None:
                if i[n.feature] <= n.feature_value:
                    n = n.left
                else:
                    n = n.right
            predictions.append(n.target)
        return predictions

    def __str__(self):
        features = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color Intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

        k_vytisteni = []
        k_vytisteni.append(self.root)
        res = ""
        while(len(k_vytisteni) > 0):
            act = k_vytisteni.pop()
            mezery = act.depth * 4 * " "
            if(act.feature is not None):
                res += mezery +  "***{} / {} / {} / {}".format(features[act.feature],act.feature_value, len(act.indices), act.criterion) + "\n"
            else:
                res += (mezery + "*" + str(act.target)) + "\n"
            if(act.right is not None):
                k_vytisteni.append(act.right)
            if act.left is not None:
                k_vytisteni.append(act.left)
        return res



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


    # Set random seed
    np.random.seed(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    tree=Tree(train_data, train_target)
    tree.build()




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

    # TODO: Finally, measure the training and testing accuracy.
    train_prediction = tree.predict(train_data)
    test_prediction = tree.predict(test_data)
    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_prediction)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_prediction)


    print("Train acc: {:.1f}%".format(100 * train_accuracy))
    print("Test acc: {:.1f}%".format(100 * test_accuracy))
    #print(tree)

