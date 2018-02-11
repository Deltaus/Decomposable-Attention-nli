#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:23:16 2018

@author: deltau
"""

class Config(object):
    
    #LEN_IN_BATCH = [20,50]
    BATCH_SIZE = 96
    TIMESTEPS = 50
    DROPOUT_RATE = 0.2
    LEARNING_RATE = {'villa':0.05, 'intra':0.025}
    NUM_LSTM_LAYERS = 2
    HIDDEN_SIZE = 200
    NUM_OF_VOCAB = 10000
    MAX_GRAD_NORM = 5
    MIN_OCCUR_NUM = 5
    MAX_OCCUR_NUM = 100
    NUM_OF_EPOCH = 30
    keep_prob = 0.5