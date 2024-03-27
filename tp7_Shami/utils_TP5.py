import numpy as np
from tensorflow import keras
import gensim

import json

def write_vectors_proj_format(m: keras.Model, text_vectorizer: keras.layers.TextVectorization):
   """
    écrit deux fichiers dans le format attendu par le projector de tensorflow
   :param m: un modèle avec une couche nommée "emb"
   :param text_vectorizer: la couche TextVectorization associée au modèle
   :return:
   """

   vecs = m.get_layer("emb").weights[0].numpy()
   with open("./dims.tsv", "w") as dims:
       with open("./labels.tsv", "w") as labels:
           for word, row in zip(text_vectorizer.get_vocabulary(), vecs):
               dims.write("\t".join([f"{x}" for x in row]) + "\n")
               if word == "":
                   word = "[PAD]"
               labels.write(word + "\n")


def write_vectors_w2v_format(m: keras.Model, text_vectorizer: keras.layers.TextVectorization, name="vectors"):
    """
    écrit un fichier .vec (format texte de w2v, lisible notamment par gensim).
    :param m: un modèle avec une couche nommée "emb"
    :param text_vectorizer: la couche TextVectorization associée au modèle
    :param name: nom du fichier ( sans extension qui sera .vec )
    :return:
    """
    vecs = m.get_layer("emb").weights[0].numpy()
    with open(f"./{name}.vec", "w") as vec_file:
        nb_w, nb_dim = vecs.shape
        vec_file.write(f"{nb_w - 1} {nb_dim}\n")
        for i, (word, row) in enumerate(zip(text_vectorizer.get_vocabulary(), vecs)):
            if i > 0:
                dims = " ".join([f"{x}" for x in row])
                vec_file.write(word + " " + dims + "\n")

def load_w2v(file: str, tv: keras.layers.TextVectorization):
    m = gensim.models.keyedvectors.load_word2vec_format(file)
    emb_size = m[0].shape[0]
    voc_size = tv.vocabulary_size()
    result = np.zeros((voc_size, emb_size))
    for i, w in enumerate(tv.get_vocabulary()):
        if w in m:
            result[i] = m[w]
    return result

def save_text_vectorization(tv: keras.layers.TextVectorization, file: str):
    config = tv.get_config()
    vocab = tv.get_vocabulary()
    with open(file, "w") as fout:
        json.dump((config, vocab), fout)

def load_text_vectorization(file: str) -> keras.layers.TextVectorization:
    with open(file, "r") as f:
        config, vocab = json.load(f)
    tv = keras.layers.TextVectorization.from_config(config)
    tv.set_vocabulary(vocab)
    return tv






