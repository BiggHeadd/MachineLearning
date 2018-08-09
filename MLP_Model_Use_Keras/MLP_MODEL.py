# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:19:25 2018

@author: 佛山
"""

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping
import os

def MLPClassifier(train_x, train_y, test_x, test_y, index, hidden_layer_size, max_iters, alphas):
    
    test_y = np_utils.to_categorical(test_y, 31)
    train_y = np_utils.to_categorical(train_y, 31)
    os.makedirs("saved_model/"+str(index))
    weights_filepath = u"saved_model/"+str(index)+'/-epoch-{epoch:02d}-val_acc_{val_acc:.4f}.hdf5'

    
    model = Sequential()
    
    model.add(Dense(units=hidden_layer_size[0], activation="sigmoid", input_dim=176))   
    model.add(Dense(units=hidden_layer_size[1], activation="sigmoid")) 
    model.add(Dense(units=31, activation='softmax'))
    
    adam = optimizers.Adam(lr=alphas)
    model.compile(optimizer=adam, metrics=['accuracy'], loss='categorical_crossentropy')
    
    model.fit(train_x, train_y, epochs=max_iters, validation_data=(test_x, test_y),
                    callbacks=[ModelCheckpoint(weights_filepath, monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='max')])

    print(model.summary)
    return model
    