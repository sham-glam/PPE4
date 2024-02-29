import numpy as np
import matplotlib.pyplot as plt
import pandas
import torch



def Model1():
    inputs = keras.layers.Input(shape=(7,), name="entrée")
    neuron = keras.layers.Dense(units=2,activation="relu", use_bias=True, name="couche1")(inputs)
    model = keras.Model(inputs=inputs, outputs=neuron)
    model.compile(optimizer="sgd", loss='mean_squared_error', metrics=["accuracy"])
    return model


MyData=pandas.read_csv('notes.csv')
MyData.pop('Eleves')
Matieres = list(MyData.keys())
print(Matieres)

Notes=MyData.to_numpy()
print(Notes)


## Avec PyTorch ##
Notes_pt=torch.tensor(Notes, dtype=torch.float64)
# defintion de la fonction d'activation ReLU
act_func_relu=torch.nn.ReLU()
# definition de la couche dense et de ses parametres
act_func_relu=torch.nn.ReLU()

# definition de la couche dense et de ses parametres
lin_filter=torch.nn.Linear(7, 2, bias=True)
lin_filter.weight.data=torch.tensor([[0.2,0.2,0.2,0,0,0.2,0.2],[0.0,0.0,0.0,0.5,0.5,0., 0.]], dtype=torch.float64)
lin_filter.bias.data=torch.tensor([-10,-10], dtype=torch.float64)

filtered_data=act_func_relu(lin_filter(Notes_pt[:,:]))

# results
print(filtered_data)


#################### Redo with Keras #####################
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

print("Nous allons faire la même chose avec Keras")

## call def Model1
model = Model1()
# Definition poids et biais pour couche dense
weights = np.array([[0.2,0.2,0.2,0,0,0.2,0.2],[0.0,0.0,0.0,0.5,0.5,0.,0.]], dtype=np.float64)
bias = np.array([-10, -10], dtype=np.float64)

model.set_weights([weights.T, bias])
print(model.summary())

keras_filtered_data = model.predict(Notes[:,:])
print(keras_filtered_data)
print(filtered_data)

