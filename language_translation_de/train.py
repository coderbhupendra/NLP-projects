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
import rnn_encoder_decoder as rnned

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
with open("data_LM/vocabe.mdl", "rb") as model:
			[s2idx, idx2s] = pickle.load(model)
with open("data_LM/vocabg.mdl", "rb") as model:
			[t2idx, idx2t] = pickle.load(model)
	

# Get embeddings for the source and the target phrase pairs

#change diretly read Wemb 
sEmbeddings = np.load('data_LM/Wembe.npy')
tEmbeddings = np.load('data_LM/Wembg.npy')
sEmbeddings=np.array(sEmbeddings,dtype='float32')
tEmbeddings=np.array(tEmbeddings,dtype='float32')
print "Loaded embeddings for source and target  embeddings"
print "loaded vacaburaly for german and english "

# Now, read the phrase table and get the phrase pairs for training
start = time.time()

with open("data/data_no.mdl", "rb") as model:
		data_noe= pickle.load(model)  
			
with open("/scratch/intern/language_translation_de/data/data_no.mdl", "rb") as model:
		data_nog= pickle.load(model)     
			
print "loaded data_no.mdl " ,time.time() - start, " sec" 


 


# Create the training and the dev partitions
#train, dev = getPartitions(phrasePairs, s['seed'])

tVocSize = len(t2idx)


# RNNED Parameters from a pre-trained run
rParameters = None
# If specified, load pre-trained parameters for the RNNED
if modelFile is not None:
	start = time.time()
	rParameters = loadModel([s2idx, t2idx, sEmbeddings, tEmbeddings])
	print "Done loading pickled parameters : ", time.time() - start, "s"

start = time.time()
rnn = rnned.RNNED(nh=nhidden, nc=tVocSize, de=emb_dimension, model=rParameters)
print "Done compiling theano functions : ", time.time() - start, "s"

# Training
best_dev_nll = np.inf
clr = lr
for e in xrange(nepochs):
	ce= e
	tic = time.time()
	phrasePairs = []
	

	s=int(len(data_nog)*.8)
	e=len(data_nog)
	savefreq=0
	#2600 3500
	for i in xrange(s):
		
		start = time.time()	
		#if ((not 10000 in data_nog[i])or(not 10000 in data_noe[i]))  :
		sPhrase = sEmbeddings[data_noe[i]]
		tPhrase = tEmbeddings[data_nog[i]]
		if np.sum(sPhrase) == 0 or np.sum(tPhrase) == 0:
			continue		
		phrasePairs.append((sPhrase,tPhrase,np.array(data_nog[i],dtype='int32')))	
			
		if ((len(phrasePairs) % 100==0) & (i>0)):
			savefreq=savefreq+1
			cost=0
			shuffle(phrasePairs, seed)
			#print  (time.time()-start) ,"sec, to make 100 sentences"
			#for idx,batch in enumerate(minibatch(phrasePairs, bs)):
			start = time.time()
			cost=rnn.train(phrasePairs, clr)
			print (time.time()-start) ," sec,to train :" ,i,"/",e	,"  sentences with cost :" ,cost 
			del phrasePairs[:]
			phrasePairs = []
			#print "preparing new 100 sentences"	
	
		if (savefreq==100):
			savefreq=0
			saveModel( s2idx, t2idx, sEmbeddings, tEmbeddings, rnn)


	

	print '[learning] epoch', e,  '>> completed in', time.time() - tic, '(sec) <<'
	sys.stdout.flush()

	# Get the average NLL For the validation set
	tic = time.time()
	phrasePairs = []
	dev_nll=0
	for i in xrange(s,e):
		
		#if ((not 10000 in data_nog[i])or(not 10000 in data_noe[i]))  :

		sPhrase = sEmbeddings[data_noe[i]]
		tPhrase = tEmbeddings[data_nog[i]]
		# Don't include phrases that contain only OOVs
		if np.sum(sPhrase) == 0 or np.sum(tPhrase) == 0:
			continue
		phrasePairs.append((sPhrase,tPhrase,np.array(data_nog[i],dtype='int32')))	

		
		if ((len(phrasePairs) % 100==0) & (i>0)):
		
			shuffle(phrasePairs, seed)
			start=time.time()
			#for idx ,batch in enumerate(minibatch(phrasePairs, bs)):
			dev_nlls = rnn.test(batch)
			dev_nll=dev_nll+np.mean(dev_nlls)
			print (time.time()-start) ,"sec,to test " ,i,"/",(e-s) ,"  sentences " 	
			del phrasePairs[:]
			phrasePairs = []
	
		
		
	dev_nll=((dev_nll*100)/(e-s))
	print '[dev-nll]', dev_nll, "(NEW BEST)" if dev_nll < best_dev_nll else "", "completed in", time.time() - tic, '(sec)'
	sys.stdout.flush()
	if dev_nll < best_dev_nll:
		best_dev_nll = dev_nll
		be = e
		saveModel( s2idx, t2idx, sEmbeddings, tEmbeddings, rnn)

	# Decay learning rate if there's no improvement in 3 epochs
	if abs(be - ce) >= 3: clr *= 0.5
	if clr < 1e-5: break

print '[BEST DEV-NLL]', best_dev_nll
print '[FINAL-LEARNING-RATE]', clr
