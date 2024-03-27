# imports utiles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from collections import Counter



def load_data(path="./corpus6.txt"):
    data = []
    with open(path) as file:
        for i, line in enumerate(file):
            if i < 600:
                label, text = line.strip().split(" ",1)
                instance = {"label":label, "text": text}
                data.append(instance)
    return data[:400], data[400:500], data[500:600]


def get_textvectorizer_3bigrams(vocab = ["t h", "e n", "l e"]):
    text_vectorizer = keras.layers.TextVectorization(
        max_tokens=4, # unk+3
        standardize="lower_and_strip_punctuation",
        split="character",
        ngrams=(2,),
        vocabulary=vocab,
        output_mode="count")
    return text_vectorizer


def PerceptronModelSparseCategorical(tv: keras.layers.TextVectorization, langues: list[str]):
    inputs = keras.layers.Input(shape=(tv.vocabulary_size(),), name="tokens_count")
    norm = keras.layers.Normalization(name="normalizer")(inputs)
    hidden = keras.layers.Dense(2*len(langues), activation="relu", use_bias=True, name="hidden")(norm)
    neuron = keras.layers.Dense(len(langues), activation="softmax", use_bias=True, name="sortie")(hidden)
    model = keras.Model(inputs=inputs, outputs=neuron)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return model

def adapt_model(model, data):
    normalization_layer = model.get_layer("normalizer")
    normalization_layer.adapt(data)
