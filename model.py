
from __future__ import absolute_import, division, print_function

import argparse
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
#from keras.models import load_model
from PIL import Image
from scipy.ndimage import zoom
from make_dataset import MakeDataset
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


#from tensorflow.keras.optimizers import SGD

#Trying to build a Convolutional Neural Net from scratch with tensorflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict script')    
    parser.add_argument('-evaluate', default=False, help='Test the model')
    parser.add_argument('-restore', default=None, help='Checkpoint path to restore training')
    args = parser.parse_args()



def save_img(ar):
    #ar = (ar*255).astype('uint8')
    name = time.time()#np.random.rand(1)
    Image.fromarray(ar).save("/home/homeGlobal/lmaehler/Documents/aa/"+str(name)+"_image.jpg")





evaluate_param = args.evaluate
restore_param = args.restore


def load_data():

    ds = MakeDataset()  
    imgs = ds.read_all(7)
    
    #print(len(imgs[0])," 333333333333333333333333333333")
    #print(len(imgs[1])," 333333333333333333333333333333")
    #print(len(imgs[2])," 333333333333333333333333333333")

    train_images = imgs[0]
    train_images = np.concatenate((train_images,imgs[1]),axis = 0)
    train_images = np.concatenate((train_images,imgs[2]), axis = 0)

    ###this leaves me with only one color channel.
    #train_images = train_images[:,:,:,0]

    #train_images = train_images[:,200:300,200:300]
    #train_images = np.transpose(train_images,(0,3,1,2))
    ###

    train_labels = np.zeros([train_images.shape[0],3])

    train_labels[:imgs[0].shape[0]]=[1,0,0]
    train_labels[imgs[0].shape[0]:(imgs[0].shape[0]+imgs[1].shape[0])]=[0,1,0]
    train_labels[(imgs[0].shape[0]+imgs[1].shape[0]):]=[0,0,1]


    #(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    #print(train_labels.shape)
    
    class_names = ['rock', 'paper', 'scissors']


    #train_images = train_images.astype(np.float32) / 255.0
    #train_images = np.asarray([zoom(e,(0.5,0.5,1)) for e in train_images])
 
 
    train_images = np.asarray([np.asarray(Image.fromarray(e).resize((int(train_images[0].shape[0]/2), int(train_images[0].shape[1]/2)))) for e in train_images])
    
    train_images, train_labels = shuffle(train_images, train_labels)


    
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_labels = keras.utils.to_categorical(train_labels)
    train_images = train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2],1)
    print(train_images.shape, "       fashion mnist shape")
    train_images = train_images[:int(train_images.shape[0]/3)]
    train_labels = train_labels[:int(train_labels.shape[0]/3)]
    train_images = np.asarray([zoom(e,(2,2,1)) for e in train_images])
    #train_images = np.asarray([zoom(e,(0.5,0.5,1)) for e in train_images])
    print(train_images.shape, "       fashion mnist shape")
    '''

    return [train_images,train_labels, class_names]



def setup_NN(input_shape):

    
    model = keras.Sequential([keras.layers.Convolution2D(8, (3,3), activation = keras.activations.relu,input_shape=input_shape),
                        keras.layers.Convolution2D(8, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(16, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(16, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(32, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(32, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.Convolution2D(64, (3,3), activation = keras.activations.relu),
                        keras.layers.MaxPooling2D(pool_size = (2,2)),
                        keras.layers.Flatten(),#input_shape =(28,28)),
                        keras.layers.Dense(60, activation = keras.activations.relu),
                        keras.layers.Dense(3, activation = keras.activations.softmax)])


    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    """

    return model



def train(restore_param = None):

    
    train_data = load_data()
    train_images = train_data[0]
    train_labels = train_data[1]
    class_names = train_data[2]


    #print(train_images.shape)
    #print(train_labels.shape, "     labels")


    #print(train_images[600])



    if not restore_param is None:
        print("\n\n\nresuming from "+restore_param+".\n\n\n")
        #model = load_from_checkpoint(restore_param, train_images.shape[1:])
        model = tf.keras.models.load_model(restore_param)

    else:

        model = setup_NN(train_images.shape[1:])
    

    print(model.summary())


    #opt = SGD(lr=0.1)
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])

    #max epochs should be 87
    model.fit(train_images[:], train_labels[:], batch_size=16, epochs=35)

    #test_loss, test_acc = model.evaluate(test_images, test_labels)

    #print('Test accuracy:', test_acc)

    model.save("./checkpoints/checkpoint.h5")




def load_from_checkpoint(path, input_shape):
    model = setup_NN(input_shape)
    model.load_weights(path)
    print("model loaded from " + path)
    return model



def evaluate(restore_param):

    eval_data = load_data()
    eval_images = eval_data[0][:3]
    eval_labels = eval_data[1][:3]
    class_names = eval_data[2]

    print("\n\n evaluating... \n\n")
    model = load_from_checkpoint(restore_param, eval_images.shape[1:])
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])
    print("\n\n checkpoint loaded... \n\n")
    
    #evaluating
    #results = model.evaluate(eval_images, eval_labels, batch_size=16)
    #print(results)

    for e in eval_images:
        save_img(e)
    predictions = model.predict(eval_images)
    
    print(class_names)
    print("predictions: {}".format([[np.round(a) for a in e] for e in predictions]))


if evaluate_param == 'True' or evaluate_param == 'true':
    evaluate(restore_param = restore_param)
else:
    train(restore_param)






