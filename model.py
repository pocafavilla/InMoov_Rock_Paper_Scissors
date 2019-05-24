
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
from make_dataset import MakeDataset

def sv(ar):
    name = np.random.rand(1)
    Image.fromarray(ar).save(str(name)+"_image.jpg")

#fashion_mnist = keras.datasets.fashion_mnist

ds = MakeDataset()
imgs = ds.read_all(4)

train_images = imgs[0]
train_images = np.concatenate((train_images,imgs[1]),axis = 0)
train_images = np.concatenate((train_images,imgs[2]), axis = 0)

###this leaves me with only one color channel.
#train_images = train_images[:,:,:,0]
print(train_images.shape)

#train_images = train_images[:,200:300,200:300]
#train_images = np.transpose(train_images,(0,3,1,2))
print(train_images.shape)
###

train_labels = np.zeros([train_images.shape[0],3])

train_labels[:imgs[0].shape[0]]=[1,0,0]
train_labels[imgs[0].shape[0]:(imgs[0].shape[0]+imgs[1].shape[0])]=[0,1,0]
train_labels[(imgs[0].shape[0]+imgs[1].shape[0]):]=[0,0,1]

print(train_labels.shape)

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(train_labels.shape)

class_names = ['rock', 'paper', 'scissors']

print(train_images.dtype)
train_images = train_images.astype(np.float32) / 255.0

print(train_images.dtype)

model = keras.Sequential([keras.layers.Convolution2D(16, (3,3), activation = keras.activations.relu,input_shape=train_images.shape[1:]),
                        keras.layers.Convolution2D(16, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(32, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(32, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Flatten(),#input_shape =(28,28)),
                        keras.layers.Dense(10, activation = keras.activations.relu),
                        keras.layers.Dense(3, activation = keras.activations.softmax)])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images[:], train_labels[:], epochs=40)

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('Test accuracy:', test_acc)

















