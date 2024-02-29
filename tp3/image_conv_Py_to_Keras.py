import torch
import torch.nn as nn
import torch.nn.functional as F


import tensorflow as tf
from tensorflow import keras
from keras import layers

from PIL import Image
# from torch.torch_version import TorchVersion
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np



import matplotlib.pyplot as plt
import numpy as np


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D




def KerasModel1():
    inputs = tf.keras.layers.Input(shape=(None, None, 1), name="input")
    # Add Conv2D layer
    conv_layer = Conv2D(filters=1, kernel_size=(3, 3), activation="relu", use_bias=True, name="conv_layer")(inputs)
    model = keras.Model(inputs=inputs, outputs=conv_layer)
    model.compile(optimizer="sgd", loss='mean_squared_error', metrics=["accuracy"])
    
    
    return model



class PyTorchCNN(nn.Module):
    
    def __init__(self):
        
        super(PyTorchCNN, self).__init__()
        
        #Convolution/ReLU/MaxPooling layers definition
        self.conv1=nn.Conv2d(1,2, kernel_size=2, stride=1, padding=1) # 1 to 2 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 to 16x16 pixels
        self.conv2=nn.Conv2d(2,4, kernel_size=2, stride=1, padding=1) # 2 to 4 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 to 8x8 pixels
        self.conv3=nn.Conv2d(4,8, kernel_size=2, stride=1, padding=1) # 4 to 8 channels
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 to 4x4 pixels
        
        # Dense layers definition
        self.fc1=nn.Linear(8*4*4, 32)
        self.fc2=nn.Linear(32,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x= self.pool1(x)
        x= F.relu(self.conv2(x))
        x= self.pool2(x)    
        x= F.relu(self.conv3(x))
        x= self.pool3(x)
        x= x.view(-1, 8*4*4)
        x= F.relu(self.fc1(x))
        x= self.fc2(x)
        
        return x
        
        

def KerasCNN():
    model = Sequential()
    model.add(Conv2D(2, kernel_size=(2, 2), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(4, kernel_size=(2, 2), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(8, kernel_size=(2, 2), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    # Dense layers definition
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()  
    
    return model
        
  
def get_with_PyTorch():
    image = Image.open("car.jpg").convert('L')  # 'L' mode converts the image to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.ToTensor()])
    # Apply transformations to the image
    image = transform(image)
    image = image.unsqueeze(0)
    # apply filters
    conv_filter=torch.nn.Conv2d(1, 2,3, bias=True)
    conv_filter.weight.data=torch.tensor([[[[-1.,0.,1.], [-1.,0.,1.], [-1.,0.,1.]]]])/3.
    conv_filter.bias.data=torch.tensor([-0.1])
    filtered_image=conv_filter(image)
    processed_image_np = filtered_image.squeeze(0).detach().numpy()
    # Visualize the processed image
    plt.imshow(np.transpose(processed_image_np, (1, 2, 0)), cmap='gray')
    plt.title('PyTorch Processed Image')
    plt.axis('off')
    plt.show()
    plt.close()
    
    
    
## Keras     
def get_with_Keras():
    # Load the image
    img = image.load_img("car.jpg", target_size=(28, 28), color_mode="grayscale")    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    
    model = KerasModel1()
    weights = np.array([[[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]]], dtype=np.float32) / 3.
    bias = np.array([-0.1], dtype=np.float32)
    model.layers[1].set_weights([weights.T, bias])
    filtered_image = model.predict(img_array)
    # Visualize the processed image
    plt.imshow(np.squeeze(filtered_image), cmap='gray',vmin=0, vmax=1)
    plt.title('Keras Processed Image')
    plt.axis('off')
    plt.show()
    plt.close()
        
        
        
        
def mnist_transform_PyTorch():
    transform = ToTensor()
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    X, y = mnist_dataset[11] # doit donner 6
    X = X.unsqueeze(0)  # Add batch dimension
    PyTorchCNN_model = PyTorchCNN()
    pred = PyTorchCNN_model(X)
    predicted_class = torch.argmax(pred)
    image = X.squeeze(0).numpy()

    # Visualize
    plt.imshow(image.squeeze(0), cmap='gray')  # Squeeze the batch dimension for visualization
    plt.title(f'Predicted with Pytorch Model: {predicted_class}')
    plt.axis('off')
    plt.show()
    plt.close()
    
    
    
    
    

def mnist_transform_Keras():
     # Load MNIST dataset
    (X_train, y_train), (_, _) = mnist.load_data()
    X = X_train[0]  # should give 6
    X = np.expand_dims(X, axis=0)  # Add batch dimension

    # Normalize input
    X = X.astype('float32') / 255

    # Load KerasCNN model
    kerasCNN_model = KerasCNN()

    # Make prediction
    pred = kerasCNN_model.predict(X)
    predicted_class = np.argmax(pred)

    # Visualize
    plt.imshow(X.squeeze(0), cmap='gray')  # Squeeze the batch dimension for visualization
    plt.title(f'Predicted with Keras Model: {predicted_class}')
    plt.axis('off')
    plt.show()
    plt.close()
    
    
    

def main():
    
    # get_with_PyTorch()
    # get_with_Keras()
    
    mnist_transform_PyTorch()
    mnist_transform_Keras()
       

if __name__ == '__main__':    
    main()
    
    
    