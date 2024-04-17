import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# %env TF_FORCE_GPU_ALLOW_GROWTH=true
import tensorflow as tf
import keras
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

def load_data(normalise=False):
    data = []
    with open("../corpus.txt") as file:
        for line in file:
            label, text = line.strip().split(" ",1)
            n_th = text.count("th")
            n_en = text.count("en")
            l = len(text)
            instance = {"label":label, "th": n_th / l, "en": n_en /l}
            data.append(instance)
    if normalise:
        normalise_data(data)
    return data

def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
            for d in dataset:
                d[k] = (d[k]- mean) / std


def perceptron_str(instance, params) -> str:
    activation = params['th'] * instance['th'] + params['en'] * instance['en']
    z = 1/(1 + np.exp(-activation))  # sigmoïde
    return "eng" if z > 0.5 else "deu" 

def perceptron_float(instance, params) -> float:
    activation = params['th'] * instance['th'] + params['en'] * instance['en']
    z = 1/(1 + np.exp(-activation))  # sigmoïde
    return z


def build_space(dataset, mini=0.1,maxi=1):
    # génération d'une grille pour couvrir l'ensemble des paramètres possibles
    # x -> poids de EN
    # y -> poids de TH
    X = np.arange(mini, maxi, (maxi-mini)/20)
    Y = np.arange(mini, maxi, (maxi-mini)/20)
    X, Y = np.meshgrid(X, Y)
    total = len(dataset)
    
    # fonction qui évalue un couple de paramètres
    def evaluation_acc(x,y)-> float:
        ok = 0
        params = {'en':x, 'th':y}
        for inst in dataset:
            if perceptron_str(inst, params) == inst['label']:
                ok += 1.0
        return ok / total
    
    
    def evaluation_mse(x,y)-> float:
        mse = 0
        params = {'en':x, 'th':y}
        for inst in dataset:
            r = perceptron_float(inst, params) 
            gold = 1.0 if inst['label'] == 'eng' else 0.0
            mse += (r - gold)**2
        return mse / total
    
    
    # On applique cette fonction à toute la grille 
    # Z = toutes les exactitudes à différents endroits de l'espace
    Z_acc = np.vectorize(evaluation_acc)(X,Y)       
    Z_mse = np.vectorize(evaluation_mse)(X,Y)
    return X, Y, Z_acc, Z_mse


from matplotlib import cm
def plot_3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z, alpha=0.8, cmap=cm.coolwarm)
    ax.set_xlabel('W EN')
    ax.set_ylabel('W TH')
    plt.show()

def plot_3d_sgd(X,Y,Z,Wx,Wy, Wz):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z, alpha=0.8, cmap=cm.coolwarm)
    ax.plot(Wx, Wy, Wz, 'b.', alpha=0.5)
    ax.set_xlabel('W EN')
    ax.set_ylabel('W TH')
    plt.show()


def plot_density_and_sgd(X, Y, Z, Wx, Wy):
    fig = plt.figure()
    ax = fig.add_subplot()
    CS = ax.contour(X,Y,Z, alpha=1, cmap=cm.coolwarm, levels=np.arange(np.min(Z), np.max(Z),(np.max(Z) - np.min(Z))/10))
    ax.plot(Wx, Wy, '.')
    ax.set_xlabel('W EN')
    ax.set_ylabel('W TH')
    ax.clabel(CS, fontsize=9, inline=True)
    plt.show()

def PerceptronModel1():
    inputs = keras.layers.Input(shape=(2,), name="entrée")
    neuron = keras.layers.Dense(1, activation="sigmoid", use_bias=False, name="couche1")(inputs)
    model = keras.Model(inputs=inputs, outputs=neuron)
    model.compile(optimizer="sgd", loss='mean_squared_error', metrics=["accuracy"])
    return model

def split_dataset(dataset):
    rest, test = train_test_split(dataset, test_size=0.1, shuffle=True)
    train, valid = train_test_split(rest, train_size=0.9, shuffle=False)
    return train, valid, test


def dataset_to_tensors(dataset):
    X = np.array([[d['th'], d['en']] for d in dataset])
    Y = np.array([0.0 if d['label'] == 'deu' else 1.0 for d in dataset])
    return X, Y


def watch_sgd(model: keras.Model, X,Y, test_data, epochs=10):
    eval_data = dataset_to_tensors(test_data)
    weights = [model.get_weights()[0]]
    losses = []
    evals = [model.evaluate(*eval_data)]
    for _ in range(epochs):
        log = model.fit(X,Y,epochs=1, batch_size=4)
        weights.append(model.get_weights()[0])
        losses.extend(log.history['loss'])
        evals.append(model.evaluate(*eval_data))
    return np.array(weights), losses, evals


def w_to_xy(weights):
    x = []
    y = []
    for row in weights:
        x.extend(row[1])
        y.extend(row[0])
    return x, y

