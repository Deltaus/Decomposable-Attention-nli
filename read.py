#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:25:16 2018

@author: deltau
"""

from nltk.tokenize import word_tokenize
from embedding import word_embedding as embed_vect
from collections import Counter
#from config import Config


def delete_bad(file_path):
    
    raw_pairs = list()
    with open(file_path,'r') as jsonl_file:
        for jsl in jsonl_file:
            raw_pairs.append(eval(jsl))
            
    bad_indexes = list()
    for raw in raw_pairs:
        if raw['gold_label'] == '-':
            bad_indexes.append(raw_pairs.index(raw))
    for ind in bad_indexes:
        del raw_pairs[ind]
        
    return raw_pairs

def get_sentences(raw_pairs):
    
    sentence1s = list()
    sentence2s = list()
    labels = list()
    for raw in raw_pairs:
        sentence1s.append(raw['sentence1'].lower())
        sentence2s.append(raw['sentence2'].lower())
        labels.append(raw['gold_label'])
    
    #print sentence1s
    s1 = [word_tokenize(sentence) for sentence in sentence1s]  #word list of sentence1
    #print s1
    s2 = [word_tokenize(sentence) for sentence in sentence2s]  #word list of sentence2
    
    return s1, s2, labels

def pad_sentence(token_list, pad_length, pad_id):
    
    padding = [pad_id] * (pad_length - len(token_list))
    padded_list = token_list + padding
    
    return padded_list   #a padded list of a single sentence


def preprocess_data(file_path):
    
    raw_pairs = delete_bad(file_path)
    s1, s2, labels = get_sentences(raw_pairs) #s1: word list of lists of sentence1 / s2: ... / labels: list of labels

    categories = ["neutral", "entailment", "contradiction"]
    lab = list()
    for l in labels:
        onehot = [0,0,0]
        ind = categories.index(l)
        onehot[ind] = 1
        lab.append(onehot)       
    labels = lab
    
    #vocab & embedding processing
    sentences = list()
    for x in s1:
        sentences.append(x)
    for x in s2:
        sentences.append(x)
    
    sentences, embeddings = embed_vect(sentences)
    vocab = Counter()
    for sentence in sentences:
        vocab.update(sentence)
    all_word_list = [x for x, y in vocab.iteritems()]# + ['<unk>', '<pad>']
    all_word_dict = dict(zip(all_word_list, xrange(len(all_word_list))))  # word:id
    for word in all_word_dict:
        all_word_dict[word] += 2
    all_word_dict['<unk>'] = 1
    all_word_dict['<pad>'] = 0
    all_word_dict = sorted(all_word_dict.items(), key=lambda item:item[1])

    #train_data
    sent_pairs = list(zip(s1,s2,labels))
    
    return sent_pairs, all_word_dict, embeddings











