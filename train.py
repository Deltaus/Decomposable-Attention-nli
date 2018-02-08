#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:24:54 2018

@author: deltau
"""

import random
import read
import tensorflow as tf
from config import Config



def run_epoch(session, model, data, train_op, iters, embed=None, output_log=False):
    
    total_costs = 0.0
    acc = 0
    iteration = iters
    embedding = embed
    for i in range(iteration):
        one_batch = batch(data, Config.BATCH_SIZE)
        prem = [x[0] for x in one_batch]
        hypo = [x[1] for x in one_batch]
        labels = [x[2] for x in one_batch]       
        prems = list()
        for sent in prem:
            words = list()
            for word in sent:
                words.append(tf.nn.embedding_lookup(embedding, word))
            prems.append(words)
        hypos = list()
        for sent in hypo:
            words = list()
            for word in sent:
                words.append(tf.nn.embedding_lookup(embedding, word))
            hypos.append(words)
        
        cost, acc, _ = session.run([model.cost, model.accuracy, model.train_op],feed_dict={model.a:prems, model.b:hypos, model.labels:labels})
        total_costs += cost    

        if output_log and i % 100 == 0:
            print 'Acc %.3f' % acc
            
    return acc

def batch(data, batch_size):
    
    one_batch = random.sample(data, batch_size)
    prem = [x[0] for x in one_batch]
    hypo = [x[1] for x in one_batch]
    labels = [x[2] for x in one_batch]
    max_prem_len = max([len(x) for x in prem])
    max_hypo_len = max([len(x) for x in hypo])
    prem = [read.pad_sentence(x, max_prem_len, '<pad>') for x in prem]
    hypo = [read.pad_sentence(x, max_hypo_len, '<pad>') for x in hypo]
    one_batch = zip(prem,hypo, labels)
    
    return one_batch
 
