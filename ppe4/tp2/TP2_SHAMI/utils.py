import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# %env TF_FORCE_GPU_ALLOW_GROWTH=true
import tensorflow as tf
import keras
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, pair_confusion_matrix

from matplotlib import cm
import seaborn as sns


# load data
def load_data(normalise=False):
    data = []
    with open("corpus6.txt") as file:
        for line in file:
            label, text = line.strip().split(" ",1)
            # print(label)
            n_th = text.count("th")
            n_en = text.count("en")
            n_es = text.count("es")
            n_le= text.count("le")
            n_be = text.count("be")
            n_co = text.count("co")
            l = len(text)
            instance = {"label":label, "th": n_th / l, "en": n_en /l, "es": n_es / l, "le": n_le / l, "be": n_be / l, "co": n_co / l}
            data.append(instance)
    # print(set(label))
    if normalise:
        normalise_data(data)
    return data

# normalise les données
def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
            for d in dataset:
                d[k] = (d[k]- mean) / std


def perceptron_str(instance, params) -> str:
    activation = params['th'] * instance['th'] + params['en'] * instance['en'] + params['es'] * instance['es'] + params['le'] * instance['le']\
        + params['be'] * instance['be'] + params['co'] * instance['co']
    z = 1/(1 + np.exp(-activation))  # sigmoïde
    return "eng" if z > 0.5 else "deu" # modify for multi-class

# returns 
def perceptron_float(instance, params) -> float:
    activation = params['th'] * instance['th'] + params['en'] * instance['en'] + params['es'] * instance['es'] \
        + params['le'] * instance['le'] + params['be'] * instance['be'] + params['co'] * instance['co']
    z = 1/(1 + np.exp(-activation))  # sigmoïde
    return z


def encode_label(label, dataset):
    unique_labels = list(set(inst['label'] for inst in dataset))
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(unique_labels)
    # original_labels = label_encoder.inverse_transform(encoded_labels)
    # print(f'original_labels: {original_labels}')
    onehot_encoder = OneHotEncoder(categories='auto')
    encoded_labels = np.array(encoded_labels).reshape(-1, 1)
    onehot_encoder.fit(encoded_labels)
    encoded_label = label_encoder.transform([label])
    onehot_encoded_label = onehot_encoder.transform(encoded_label.reshape(-1, 1)).toarray()
    # print(f'onehot_encoded_label: {onehot_encoded_label.flatten()}')
    
    return onehot_encoded_label.flatten()

# perceptron model
def PerceptronModel1():
    inputs = keras.layers.Input(shape=(2,), name="entrée")
    neuron = keras.layers.Dense(1, activation="sigmoid", use_bias=False, name="couche1")(inputs)
    model = keras.Model(inputs=inputs, outputs=neuron)
    model.compile(optimizer="sgd", loss='mean_squared_error', metrics=["accuracy"])
    return model

# sequential model
def PerceptronModel2():
    model = keras.Sequential([
            keras.Input(shape=(6,), name="entrée"),
            keras.layers.Dense(16, activation="sigmoid", name="layer1", use_bias=True), 
            keras.layers.Dense(8, activation="sigmoid", name="layer2", use_bias=True),
            keras.layers.Dense(6, activation='softmax') # avant 'categorical_crossentropy' forcément 'softmax'
        ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    return model



# split dataset
def split_dataset(dataset):
    rest, test = train_test_split(dataset, test_size=0.1, shuffle=True) # test
    train, valid = train_test_split(rest, train_size=0.9, shuffle=False) # train, valid
    return train, valid, test


# convert dataset to tensors
def dataset_to_tensors(dataset):
    X = np.array([[d['th'], d['en'], d['es'], d['le'], d['be'], d['co']] for d in dataset])
    # Y = np.array([0.0 if d['label'] == 'deu' else 1.0 for d in dataset])
    labels = [d['label'] for d in dataset]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    return X, Y

#
    
def draw_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize='all')
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='viridis')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
