#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
from rnnforsentiment.data_process import vocab_dim, maxlen
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.models import Model
import sys

np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)


def rnnModel(n_symbols, embedding_weights):
    #define input
    sentence_indices = Input(shape=(maxlen,), dtype='int32')
    #define embedding layer
    embeddings = Embedding(input_dim=n_symbols,
              output_dim=vocab_dim,
              mask_zero=True,
              weights=[embedding_weights],
              input_length=maxlen,
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