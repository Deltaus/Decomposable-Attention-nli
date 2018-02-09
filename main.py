#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:25:39 2018

@author: deltau
"""

import tensorflow as tf
import numpy as np
from model import Model
from embedding import word_embedding as embed_vect
import read
from config import Config
from collections import Counter
import random
import train


if __name__=="__main__":
    
    path = 'snli-corpus/'
    #entire file
    file_train = 'snli_1.0_train.jsonl'
    file_valid = 'snli_1.0_dev.jsonl'
    file_test =  'snli_1.0_test.jsonl'
    #sample file
    sample_file_train = 'train_sample.jsonl'
    sample_file_valid = 'dev_sample.jsonl'
    sample_file_test = 'test_sample.jsonl'

    #train_data  labels in one-hot
    sent_pairs_train, _, embeddings_t = read.preprocess_data(path+sample_file_train)
    #vad_data
    sent_pairs_valid, _, embeddings_v = read.preprocess_data(path+sample_file_valid)
    #test_data
    sent_pairs_test, _, embeddings_e = read.preprocess_data(path+sample_file_test)
    #num of samples in each dataset
    train_sample_num = len(sent_pairs_train)
    train_iters = round(train_sample_num / Config.BATCH_SIZE)
    valid_sample_num = len(sent_pairs_valid)
    valid_iters = round(valid_sample_num / Config.BATCH_SIZE)
    test_sample_num = len(sent_pairs_test)
    test_iters = round(test_sample_num / Config.BATCH_SIZE)

    # run epoch
    tf.reset_default_graph()
    with tf.variable_scope('snli_model'):
        train_model = Model(Config, is_training=True)
    with tf.variable_scope('snli_model', reuse=True):
        eval_model = Model(Config, is_training=False)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(Config.NUM_OF_EPOCH):
            print 'In epoch: %d/%d' % (epoch + 1,Config.NUM_OF_EPOCH)
            print 'Training:'
            _, t_costs = train.run_epoch(sess, train_model, sent_pairs_train, train_iters, train_model.train_op, embed=embeddings_t, output_log=True)       
            print 'Training total costs: %.3f' % t_costs
            print 'Validating:'
            e_acc, e_costs = train.run_epoch(sess, eval_model, sent_pairs_valid, valid_iters, train_op=None, embed=embeddings_t, output_log=False)
            print 'Validate acc:%.3f, total costs: %.3f' % (e_acc,e_costs)
        
        print 'Testing:'
        test_acc, test_costs = train.run_epoch(sess, eval_model, sent_pairs_test, test_iters, train_op=None, embed=embeddings_t, output_log=False)
        print 'Test Accuracy: %.3f, Costs: %.3f' % (test_acc,test_costs)




