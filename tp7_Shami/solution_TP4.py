# imports utiles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Literal

from utils_TP4 import *

@dataclass
class ExpeConfig:
    tokenization: Literal["whitespace", "character"]
    ngrams: Optional[tuple[int]]
    voc_size: int

@dataclass
class ExpeResult:
    tokenization: Literal["whitespace", "character"]
    voc_size: int
    dev_acc: float
    test_acc: float
    num_params: int 
    vocab: list[str]


AGENDA = [
        ExpeConfig("character", None, 10),
        ExpeConfig("whitespace", None, 10),
        ExpeConfig("character", None, 100),
        ExpeConfig("character", None, 1000),
        ExpeConfig("whitespace", None, 1000),
        ExpeConfig("whitespace", None, 10000),
        ExpeConfig("whitespace", (2,), 10000),
        ExpeConfig("character", (2,), 100),
        ExpeConfig("character", (2,4), 100),
        ExpeConfig("character", (2,3,4), 100),
        ExpeConfig("character", (4,), 500),
        ]

def get_text_vectorizer_from_config(cfg: ExpeConfig) -> keras.layers.TextVectorization:
    return keras.layers.TextVectorization(
        max_tokens=cfg.voc_size,
        split=cfg.tokenization,
        ngrams=cfg.ngrams,
        standardize="lower_and_strip_punctuation",
        output_mode="count"
    )

def run_expe(cfg: ExpeConfig, train, dev, test, verbose=False) -> ExpeResult:
    langues = list(sorted(set(d['label'] for d in train)))
    langue_to_int = {l: i for i, l in enumerate(langues)}
    tv = get_text_vectorizer_from_config(cfg)
    tv.adapt([x['text'] for x in train])
    X_train = np.array([tv(d['text']) for d in train])
    Y_train = np.array([langue_to_int[d['label']] for d in train])
    X_dev = np.array([tv(d['text']) for d in dev])
    Y_dev = np.array([langue_to_int[d['label']] for d in dev])
    X_test = np.array([tv(d['text']) for d in test])
    Y_test = np.array([langue_to_int[d['label']] for d in test])
    m = PerceptronModelSparseCategorical(tv, langues)
    adapt_model(m, X_train)
    if verbose:
        m.summary()
    hist = m.fit(x=X_train, y=Y_train, epochs=100, batch_size=32, validation_data=(X_dev, Y_dev), verbose=0)
    if verbose:
        print(f"last val accuracy: {hist.history['val_accuracy'][-1]}")
    test_acc = m.evaluate(X_test, Y_test, return_dict=True, verbose=0)["accuracy"]
    return ExpeResult(
            cfg.tokenization, 
            cfg.voc_size, 
            hist.history['val_accuracy'][-1], 
            test_acc, 
            m.count_params(),
            tv.get_vocabulary()
            )

def main():
    train, dev, test = load_data()
    for cfg in AGENDA:
        result = run_expe(cfg, train, dev, test)
        print(result)
     
        
if __name__ == "__main__":
    main()
