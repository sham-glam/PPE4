import csv
import tensorflow as tf
import unicodedata
from pprint import pprint
import keras 
import pandas as pd



# lecture et stockage des 2 colonnes souhaités
def read_csv_file(tsv_file, colN1, colN2):
    with open(tsv_file) as t:
        tsv_reader = csv.reader(t, delimiter='\t')
        # get the 1st and 2nd coDlumn
        tokens = []
        morphs = []
        for line in tsv_reader:
             
            tokens.append(line[0].split(' ')[colN1])
            morphs.append(line[0].split(' ')[colN2])

    return tokens, morphs


# construction manuelle du vocabulaire 
def build_vocab(tokens):
    temp_vocab = list(map(lambda token: tf.strings.unicode_split(token, input_encoding='UTF-8'), tokens))
    flat_vocab = [item.numpy().decode('utf-8') for sublist in temp_vocab for item in sublist]
    vocab_unique = list(set(flat_vocab))  
    vocab = ['[START]', '[END]'] + vocab_unique # il déduit automatiquement '', [UNK], erreur si on les ajoute

    return vocab
    


''' create instances '''
def create_instances(data_X, data_Y, vocab):
    tv = tf.keras.layers.TextVectorization(
        output_sequence_length=48, 
        max_tokens=1000,
        output_mode='int', 
        split=None,
        vocabulary=vocab,
        standardize=None,
        ragged=False, # default
        encoding='utf-8',
        pad_to_max_tokens=True
    )
    print("X1 before split: ", data_X[:2])
    X1 = tf.strings.unicode_split(data_X, input_encoding='UTF-8') # Split X1 into characters
    print("X1: ", X1[:2])
    X1_vectorized = tv(X1)
    
    X2  = list(map(lambda x: '[START]'+ x, data_Y)) # add '[START]' to the beginning of each morpheme
    print("\n X2 before split: ", X2[:2])
    X2 = tf.strings.unicode_split(data_Y, input_encoding='UTF-8')
    print("\n X2: ", X2[:2])
    X2_vectorized = tv(X2)

    # add '[END]' to the end of each morpheme
    Y = list(map(lambda x: x + '[END]', data_Y))
    print("\nY beofore split: ",Y[:2])
    Y = tf.strings.unicode_split(data_Y, input_encoding='UTF-8')
    print("\nData Y",Y[:2])
    Y_vectorized = tv(Y)

    return X1_vectorized, X2_vectorized, Y_vectorized, tv

   

# nomalise if time
def normalise():
    string = "cooreen"
    normalized_string = unicodedata.normalize('NFKC', string)
    print("Chaîne normalisée avec NFKC:", normalized_string)



############  lecture fichiers co-NLU  ###############
def parse_conllu_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) > 6:
                    token,lemma, lemma_tag = parts[1], parts[2], parts[4]  # lemma tag is in the 5th column
                    data.append((token, lemma, lemma_tag))
    return pd.DataFrame(data, columns=['token', 'lemma', 'lemma_tag'])

# conllu vocab
def create_char_vocab(data):
    unique_chars = set()
    for column in ['token', 'lemma', 'lemma_tag']:
        all_text = ''.join(data[column].tolist())
        unique_chars.update(tf.strings.unicode_split(all_text, 'UTF-8').numpy().tolist())
    vocab = list(unique_chars)
    # vocab.extend(['[START]', '[END]', '[UNK]'])
    return vocab



''' create instances CONLL '''
def conll_instances(data_X, vocab):
    tv = tf.keras.layers.TextVectorization(
        output_sequence_length=48, 
        max_tokens=10000,
        output_mode='int', 
        split=None,
        vocabulary=vocab,
        standardize=None,
        ragged=False, # default
        encoding='utf-8',
        pad_to_max_tokens=True
    )

    X_tokens = data_X['token']
    X_lemmas = data_X['lemma']
    X_lemma_tags = data_X['lemma_tag']

    pprint(X_tokens[:100])
    pprint(X_lemmas[:100])
    pprint(X_lemma_tags[:100])

    X1 = tf.strings.unicode_split(X_tokens, input_encoding='UTF-8') # Split X1 into characters
    print("X1: ", X1[:2])
    X1_vectorized = tv(X1)

    X2  = list(map(lambda x: '[START]'+ x, X_lemmas)) # add '[START]' to the beginning of each morpheme
    X2 = tf.strings.unicode_split(X2, input_encoding='UTF-8') # Split X1 into characters
    print("X2: ", X2[:2])
    X2_vectorized = tv(X2)

    X3 = list(map(lambda x: '[START]'+ x, X_lemma_tags))
    X3 = tf.strings.unicode_split(X3, input_encoding='UTF-8') # Split X1 into characters
    print("X3: ", X3[:2])
    X3_vectorized = tv(X3)

    Y1 = list(map(lambda x: x + '[END]', X_lemmas))
    Y1 =tf.strings.unicode_split(Y1, input_encoding='UTF-8')
    Y1_vectorized = tv(Y1)

    Y2 = list(map(lambda x: x + '[END]', X_lemma_tags))
    Y2 = tf.strings.unicode_split(Y2, input_encoding='UTF-8')
    Y2_vectorized = tv(Y2)

    return X1_vectorized, X2_vectorized, X3_vectorized, Y1_vectorized, Y2_vectorized, tv

