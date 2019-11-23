from sklearn.metrics import accuracy_score

import mnist_competition as mn
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import numpy as np
class Dataset:
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        dataset = np.load(name)
        self.data, self.target = dataset["data"].reshape([-1, 28*28]).astype(np.float), dataset["target"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

train = Dataset()


a = mn.recodex_predict(train.data)
print(accuracy_score(train.target,a))