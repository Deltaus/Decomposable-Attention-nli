#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:24:54 2018

@author: deltau
"""

import random
import read
import tensorflow as tf
import numpy as np
from config import Config



def run_epoch(session, model, data, iters, train_op=None, embed=None, output_log=False):
    
    total_costs = 0.0
    acc = 0
    iteration = int(iters)
    #embedding = embed
    for i in range(iteration):
        prems, hypos, labels, max_p, max_h = batch(data, Config.BATCH_SIZE, embed)
        #model.TIMESTEPS_A = max_p
        #model.TIMESTEPS_B = max_h
        #print max_p
        #print model.TIMESTEPS_A
        #print max_h
        #print model.TIMESTEPS_B
        if train_op is not None:
            cost, acc, _ = session.run([model.cost, model.accuracy, train_op],\
                                   feed_dict={model.a:prems, model.b:hypos, model.labels:labels})
        else:
            cost, acc = session.run([model.cost, model.accuracy],\
                                   feed_dict={model.a:prems, model.b:hypos, model.labels:labels})
            
        total_costs += cost    

        if output_log: #and i % 100 == 0:
            print 'After %d iteration(s), Acc is %.3f' % (i+1,acc)
            
    return acc, total_costs

def batch(data, batch_size, embedding):
    
    one_batch = random.sample(data, batch_size)
    prem = [x[0] for x in one_batch]
    hypo = [x[1] for x in one_batch]
    labels = [x[2] for x in one_batch]
    max_prem_len = max([len(x) for x in prem])
    max_hypo_len = max([len(x) for x in hypo])
    prem = [read.pad_sentence(x, max_prem_len, '<pad>') for x in prem]
    hypo = [read.pad_sentence(x, max_hypo_len, '<pad>') for x in hypo]
    
    prems = list()
    for sent in prem:
        words = list()
        for word in sent:
            if word == '<pad>':
                words.append(np.zeros([Config.HIDDEN_SIZE]))
            else:
                try:
                    words.append(embedding[word])
                except:
                    words.append(tf.random_normal([Config.HIDDEN_SIZE]).eval())
        prems.append(words)
    hypos = list()
    for sent in hypo:
        words = list()
        for word in sent:
            if word == '<pad>':
                words.append(np.zeros([Config.HIDDEN_SIZE]))
            else:
                try:
                    words.append(embedding[word])
                except:
                    words.append(tf.random_normal([Config.HIDDEN_SIZE]).eval())
        hypos.append(words)
        
    
    ######
    max_prem_len = Config.TIMESTEPS
    max_hypo_len = Config.TIMESTEPS
    
    return prems, hypos, labels, max_prem_len, max_hypo_len
 
