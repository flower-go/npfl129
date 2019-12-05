#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

import numpy as np

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/1920/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.data = dataset_file.read()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def create_tuples(text, n):

    text = padding(text, n)
    n = n*2 + 1
    return ["".join(a) for a in zip(*[text[i:] for i in range(n)])]

def padding(text, n):
    start  = n
    end = n
    return [" "]*n + text + [" "]*n

def convert_to_numbers(x):
    carka = "áéíóúý"
    krouzek="ů"
    hacek = "čďěňřšťž"
    if x in carka:
        return 0
    elif x in krouzek:
        return 1
    elif x in hacek:
        return 2
    else:
        return 3

def transform_data(train):



if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)


    # Load the dataset, downloading it if required
    train = Dataset()
    chars_orig= list(train.data)
    labels_orig = chars_orig


    mask = [True if i in train.LETTERS_NODIA or i in train.LETTERS_DIA else False for i in labels_orig]
    chars = list(np.array(chars_orig)[mask])
    labels = list(np.array(labels_orig)[mask])


    tuples_3 = create_tuples(chars,1)
    tuples_4 = create_tuples(chars,2)

    data_t = np.array([chars,tuples_3, tuples_4]).T

    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(data_t)

    # 3. Transform
    onehotlabels = enc.transform(data_t)

    labels_t = [convert_to_numbers(x) for x in labels]

    # TODO: Train the model.
    model = MLPClassifier(activation='relu')
    model.fit(onehotlabels,labels_t)
    # TODO: The trained model needs to be saved. All sklearn models can
    # be serialized and deserialized using the standard `pickle` module.
    # Additionally, we also compress the model.
    #
    # To save a model, open a target file for binary access, and use
    # `pickle.dump` to save the model to the opened file:
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)

# The `recodex_predict` is called during ReCodEx evaluation (there can be
# several Python sources in the submission, but exactly one should contain
# a `recodex_predict` method).
def recodex_predict(data):
    # The `data` is a `str` containing text without diacritics

    args = parser.parse_args([])

    # TODO: Predict target values for the given data.
    #
    # You should probably start by loading a model. Start by opening the model
    # file for binary read access and then use `pickle.load` to deserialize the
    # model from the stored binary data:
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)



    # TODO: Return the predictions as a diacritized `str`. It has to have
    # exactly the same length as `data` and all corresponding characters
    # must be the same, except tha may differ in diacritics.
