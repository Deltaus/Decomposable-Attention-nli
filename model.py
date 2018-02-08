#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:24:10 2018

@author: deltau
"""

import tensorflow as tf
from config import Config as conf
import numpy as np


class Model(object):
    
    def __init__(self, config, is_training, embedding_vect=None, a_step=50, b_step=50):
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.labels = tf.placeholder(tf.int32, [self.batch_size, 3])
        self.a = tf.placeholder(tf.float32)
        self.b = tf.placeholder(tf.float32)
        self.is_training = is_training
        self.embedding = embedding_vect
        self.TIMESTEPS_A = a_step
        self.TIMESTEPS_B = b_step
        
        a = tf.reshape(self.a, [self.batch_size, self.TIMESTEPS_A, self.hidden_size])
        b = tf.reshape(self.b, [self.batch_size, self.TIMESTEPS_B, self.hidden_size])
              
        #feed forward
        #tf.reset_default_graph()
        with tf.variable_scope('ip_prc'):
            a_ff = self.feed_forward_isa(a)                   # BATCH_SIZE x TIMESTEPS_A x HIDDEN_SIZE
        with tf.variable_scope('ip_prc', reuse=True):
            b_ff = self.feed_forward_isa(b)                   # BATCH_SIZE x TIMESTEPS_B x HIDDEN_SIZE
        f = tf.matmul(a_ff, b_ff, adjoint_b=True)  # BATCH_SIZE x TIMESTEPS_A x TIMESTEPS_B
        #intra attention code
        
        
        
        #attend
        #e = f if no intra attention
        softmax_row_as_tb = tf.reshape(f, [self.batch_size * self.TIMESTEPS_A, -1])         # (BATCH_SIZE x TIMESTEPS_A) x TIMESTEPS_B
        softmax_row_as_tb = tf.nn.softmax(softmax_row_as_tb)
        softmax_row_as_tb = tf.expand_dims(softmax_row_as_tb,2)                             # (BATCH_SIZE x TIMESTEPS_A) x TIMESTEPS_B x 1
        beta = tf.reduce_sum(softmax_row_as_tb * tf.tile(b, [self.TIMESTEPS_A, 1, 1]), [1]) # (BATCH_SIZE x TIMESTEPS_A) x HIDDEN_SIZE
        beta = tf.reshape(beta, [self.batch_size, -1, self.hidden_size])                    # BATCH_SIZE x TIMESTEPS_A x HIDDEN_SIZE
        #beta = tf.Variable([tf.squeeze(x) for x in tf.split(beta, self.TIMESTEPS_A, 1)],'beta')                 # each x of size BATCH_SIZE x HIDDEN_SIZE
        beta = [tf.squeeze(x) for x in tf.split(beta, self.TIMESTEPS_A, 1)]
        
        f = tf.transpose(f, [0,2,1])
        softmax_row_as_ta = tf.reshape(f, [self.batch_size * self.TIMESTEPS_B, -1])          # (BATCH_SIZE x TIMESTEPS_B) x TIMESTEPS_A
        softmax_row_as_ta = tf.nn.softmax(softmax_row_as_ta)
        softmax_row_as_ta = tf.expand_dims(softmax_row_as_ta,2)                              # (BATCH_SIZE x TIMESTEPS_B) x TIMESTEPS_A x 1
        alpha = tf.reduce_sum(softmax_row_as_ta * tf.tile(a, [self.TIMESTEPS_B, 1, 1]), [1]) # (BATCH_SIZE x TIMESTEPS_B) x HIDDEN_SIZE
        alpha = tf.reshape(alpha, [self.batch_size, -1, self.hidden_size])                   # BATCH_SIZE x TIMESTEPS_B x HIDDEN_SIZE
        #alpha = tf.Variable([tf.squeeze(x) for x in tf.split(alpha, self.TIMESTEPS_B, 1)],'alpha')               # each x of size BATCH_SIZE x HIDDEN_SIZE
        alpha = [tf.squeeze(x) for x in tf.split(alpha, self.TIMESTEPS_B, 1)]
        
        a = [tf.squeeze(x) for x in tf.split(a, self.TIMESTEPS_A, 1)]
        b = [tf.squeeze(x) for x in tf.split(b, self.TIMESTEPS_B, 1)]
        
        #compare
        a_with_beta = list()
        b_with_alpha = list()
        for word, weight in zip(a, beta):
            a_with_beta.append(tf.concat([word,weight], 1))     #  BATCH_SIZE x HIDDEN_SIZE*2  x TIMESTEPS_B        
        for word, weight in zip(b, alpha):
            b_with_alpha.append(tf.concat([word,weight], 1))
        
        a_with_beta = tf.Variable(a_with_beta,'awb')
        b_with_alpha = tf.Variable(b_with_alpha,'bwa')
        print a_with_beta.shape
        print b_with_alpha.shape
        
        with tf.variable_scope('compare'):
            tp = [tf.squeeze(x) for x in tf.split(a_with_beta,self.TIMESTEPS_A,0)]
            v1 = tf.split(self.feed_forward(tf.concat(tp,0)), self.TIMESTEPS_A, 0)
        with tf.variable_scope('compare', reuse=True):
            tp = [tf.squeeze(x) for x in tf.split(b_with_alpha,self.TIMESTEPS_B,0)]
            v2 = tf.split(self.feed_forward(tf.concat(tp,0)), self.TIMESTEPS_B, 0)
        
        #aggregate
        v1_sum = tf.add_n(v1)  # BATCH_SIZE x HIDDEN_SIZE
        v2_sum = tf.add_n(v2)  # BATCH_SIZE x HIDDEN_SIZE
        with tf.variable_scope('aggregate'):
            final = self.feed_forward(tf.concat([v1_sum,v2_sum], 1)) # BATCH_SIZE x HIDDEN_SIZE * 2
        
        softmax_w = tf.get_variable('softmax_w', [4 * self.hidden_size, 3])
        softmax_b = tf.get_variable('softmax_b', [3])
        self.categories = tf.matmul(final, softmax_w) + softmax_b
        
        _, labels = tf.nn.top_k(self.labels)
        _, categ = tf.nn.top_k(self.categories)
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.categories], [labels], [tf.ones([self.batch_size])])
        self.cost = tf.reduce_mean(loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(categ, labels), tf.float32))
        
        if is_training:
            self.lr = tf.Variable(self.config.LEARNING_RATE['villa'], trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.config.MAX_GRAD_NORM)

            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdagradOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        
    #feed forward with relu
    def feed_forward(self, input_vect):
        
        HIDDEN_UNITS = input_vect.get_shape().as_list()[1]
        print HIDDEN_UNITS
        weight_layer1 = tf.get_variable('fw1', [HIDDEN_UNITS, HIDDEN_UNITS])
        bias_layer1 = tf.get_variable('fb1', [HIDDEN_UNITS])
        weight_layer2 = tf.get_variable('fw2', [HIDDEN_UNITS, HIDDEN_UNITS])
        bias_layer2 = tf.get_variable('fb2', [HIDDEN_UNITS])

        hidden_units = tf.nn.relu(tf.matmul(input_vect, weight_layer1) + bias_layer1)
        output_vect = tf.nn.relu(tf.matmul(hidden_units, weight_layer2) + bias_layer2)
        
        return output_vect

    
    # feed forword with relu and Intra-Sentence Attention 
    # need changes
    def feed_forward_isa(self, input_vect):
        
        STEPS = input_vect.get_shape().as_list()[1]

        weight_layer1 = tf.tile(tf.expand_dims(tf.get_variable('iw1', [self.hidden_size, self.hidden_size]),0),[conf.BATCH_SIZE,1,1])
        bias_layer1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.get_variable('ib1', [self.hidden_size]),0),0),[24,50,1])
        weight_layer2 = tf.tile(tf.expand_dims(tf.get_variable('iw2', [self.hidden_size, self.hidden_size]),0),[conf.BATCH_SIZE,1,1])
        bias_layer2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.get_variable('ib2', [self.hidden_size]),0),0),[24,50,1])
        
        hidden_units = tf.nn.relu(tf.matmul(input_vect, weight_layer1) + bias_layer1)
        output_vect = tf.nn.relu(tf.matmul(hidden_units, weight_layer2) + bias_layer2)
        
        return output_vect
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        