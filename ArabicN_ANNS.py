# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:40:13 2020

@author: Karim
"""
#importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf

#importing dataset
x_train = pd.read_csv("csvTrainImages 60k x 784.csv")
y_train = pd.read_csv("csvTrainLabel 60k x 1.csv")
x_test = pd.read_csv("csvTestImages 10k x 784.csv")
y_test = pd.read_csv("csvTestLabel 10k x 1.csv")

#normalizing pixils
x_train = x_train / 255
x_test = x_test / 255

#reshaping dataset
x_train=x_train.values.reshape(-1, 784)
y_train = y_train.values

x_test = x_test.values.reshape(-1, 784)
y_test = y_test.values



#neural network
def neural_net():
    
    class myCallBacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            
            #training stops if accuracy is >= 0.988.
            if(logs.get('acc') >= 0.998):
                print("Reached 99% accuaracy cancelling training")
                self.model.stop_training = True
                
    mcallbacks = myCallBacks()
    
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.InputLayer(input_shape=(784,)),
        
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        
        ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=500,callbacks=[mcallbacks], validation_data=(x_test, y_test))

    return history.epoch, history.history['acc'][-1]
        
neural_net()    



