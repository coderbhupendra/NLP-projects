#!/usr/bin/env python

"""
Wrapper script to train an RNN-encoder-decoder model for phrase translation probabilities
"""

import os
import sys
import gzip
import time
import codecs
import random
import pickle
import operator
import argparse
import numpy as np
import theano
from collections import defaultdict
import rnn_e_d_test as rnned
theano.config.warn_float64='warn'
theano.config.floatX='float32'

# For pickle to work properly
sys.setrecursionlimit(50000)


def minibatch(l, bs):
    """
    Yield batches for mini-batch SGD

    Parameters:
        l : The list of training examples
        bs : The batch size

    Returns:
        Iterator over batches
    """
    for i in xrange(0, len(l), bs):
        yield l[i:i+bs]


def shuffle(l, seed):
    """
    Shuffles training samples (in-place)

    Parameters:
        l : The training samples
        seed : A seed for the RNG
    """
    #print "shuffleing done "
    
    random.seed(seed)
    random.shuffle(l)


def getPartitions(phrasePairs, seed):
    """
    Gets training and validation partitions (80/20) from a set of training samples

    Parameters:
        phrasePairs : The training samples
        seed : A seed for the RNG
    """
    shuffle(phrasePairs, seed)
    # 80/20 partition for train/dev
    return phrasePairs[:int(0.8 * len(phrasePairs))], phrasePairs[int(0.8 * len(phrasePairs)):]


def saveModel(sVocab, tVocab, sEmbedding, tEmbedding, rnn):
    """
    Pickles a model
    Don't pickle the entire RNN object (needs deep recursion limit and may be GPU compiled)

    Parameters:
        outDir : The output directory (created if it does not exist)
        sVocab : The source vocabulary
        tVocab : The target vocabulary
        sEmbedding : The source word embeddings
        tEmbedding : The target word embeddings
        rnn : An RNN encoder-decoder model
    """
    print "saving..."
    lParameters = [sVocab, tVocab, sEmbedding, tEmbedding]
    rParameters = rnn.getParams()
    with open("data_LM/best.mdl", "wb") as m:
        pickle.dump([[lParameters], [rParameters]], m)


def loadModel(lParams):
    """
    If a model file is specified from a previous training example
    load it to initialize the RNNED object

    Parameters:
        lParams : Language parameters (for santity check only)

    Returns:
        rParameters : The parameters of the RNNED model
    """
    lParameters = None
    rParameters = None
    # Read parameters from a pickled object
    with open("data_LM/best.mdl", "rb") as model:
        [[lParameters], [rParameters]] = pickle.load(model)
    #assert lParams == lParameters
    return rParameters

modelFile=1

# Hyperparameters

lr=0.0827 # The learning rate
bs=100 # size of the mini-batch
nhidden=500 # Size of the hidden layer
seed=324 # Seed for the random number generator
emb_dimension=100 # The dimension of the embedding
nepochs=25# The number of epochs that training is to run for
prune_t=5000 # The frequency threshold for histogram pruning of the vocab


# First process the training dataset and get the source and target vocabulary

#change the vocabs
with open("data/vocab.mdl", "rb") as model:
        [vocabe, rVocabe] = pickle.load(model)
with open("/scratch/intern/language_translation_de/data/vocab.mdl", "rb") as model:
        [vocabg, rVocabg] = pickle.load(model)      
    

# Get embeddings for the source and the target phrase pairs

#change diretly read Wemb 
#sdata = np.load('lstm_model.npz')
#sEmbeddings=sdata['Wemb.npy']
sEmbeddings=np.load('/scratch/intern/language_translation_en/ngrammodel/model/embeddings.npy')


#tdata = np.load('/scratch/intern/language_translation_de/lstm_model.npz')
#tEmbeddings=tdata['Wemb.npy']
tEmbeddings=np.load('/scratch/intern/language_translation_de/ngrammodel/model/embeddings.npy')

sEmbeddings=np.array(sEmbeddings,dtype='float32')
tEmbeddings=np.array(tEmbeddings,dtype='float32')
print "Loaded embeddings for source and target  embeddings"
print "loaded vacaburaly for german and english "

# Now, read the phrase table and get the phrase pairs for training
start = time.time()

with open("/scratch/intern/language_translation_en/data/data_noe.mdl", "rb") as model:
        data_noe= pickle.load(model)  
            
with open("/scratch/intern/language_translation_de/data/data_nog.mdl", "rb") as model:
        data_nog= pickle.load(model)     
            
print "loaded data_no.mdl " ,time.time() - start, " sec" 

 


# Create the training and the dev partitions
#train, dev = getPartitions(phrasePairs, s['seed'])

tVocSize = len(rVocabg)


# RNNED Parameters from a pre-trained run
rParameters = None
# If specified, load pre-trained parameters for the RNNED
if modelFile is not None:
    start = time.time()
    rParameters = loadModel([vocabe, vocabg, sEmbeddings, tEmbeddings])
    print "Done loading pickled parameters : ", time.time() - start, "s"

start = time.time()
rnn = rnned.RNNED(nh=nhidden, nc=tVocSize, de=emb_dimension, model=rParameters)
print "Done compiling theano functions : ", time.time() - start, "s"

# Training
for ww in range(1):

    

    phrasePairs = []
    

    s=int(len(data_nog)*.8)
    e=len(data_nog)
    
    for i in xrange(0,s,1000):
        
        if((len(data_noe[i])>=1) & (len(data_nog[i])>=1)):
            sPhrase=sEmbeddings[data_noe[i]]
            phrasePairs.append((sPhrase,len(data_nog[i])))
            translation=rnn.train(phrasePairs)
            print "correct english=============================== i, ", i
            l=[]
            print len(data_noe[i])
            for w in range (len(data_noe[i])):
                l.append(rVocabe[data_noe[i][w]])
            print ' '.join(word.encode("utf8") for word in l)
            l=[]
            print len(data_nog[i])
            print "correct german"
            for w in range (len(data_nog[i])):
                l.append(rVocabg[data_nog[i][w]])
            print ' '.join(word.encode("utf8") for word in l)
            l=[]
            print len(translation[0])
            print "translated german"
            for w in range (len(translation[0])):
                l.append(rVocabg[translation[0][w]])
            print ' '.join(word.encode("utf8") for word in l)   
            print "=====================================\n"
            del phrasePairs[:] 
            phrasePairs = []
                        

        
        
    

    print '[learning] epoch', e,  '>> completed  <<'
    sys.stdout.flush()

    