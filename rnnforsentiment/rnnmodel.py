#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
from rnnforsentiment.data_process import vocab_dim, maxlen
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.models import load_model, Model
import keras
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml

n_epoch = 1
input_length = 100
batch_size = 32


##定义网络结构
def rnnmodel(n_symbols, embedding_weights):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length,
                        trainable=False))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model

def rnnModel(n_symbols, embedding_weights):
    #define input
    sentence_indices = Input(shape=(maxlen,), dtype='int32')
    #define embedding layer
    embeddings = Embedding(input_dim=n_symbols,
              output_dim=vocab_dim,
              mask_zero=True,
              weights=[embedding_weights],
              input_length=input_length,
              trainable=False)(sentence_indices)
    #define LSTM 1
    X = LSTM(256, return_sequences=True)(embeddings)
    #Dropout
    X = Dropout(0.5)(X)
    #define LSTM 2
    X = LSTM(256)(X)
    #Dropout
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(3, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)
    return model