"""f = open('europarl-v7.de-en.en', 'rU')
c=0
a=[]
for line in f:
	print(line.strip())
	a.append(line.strip())
	c=c+1

print c
print len(a)
print a[10]
print a[2000]
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
import nltk
from collections import defaultdict

def parseCorpus(iFile, pruneThreshold):
	c=0
	f = open(iFile, 'rU')
	freq = defaultdict()
	for line in f:
	#	print line
		words = nltk.word_tokenize(line)
		for word in words:
			freq[word] = freq.get(word, 0) + 1

	"""for line in f:
		print line
		words = line.strip().split()
		print words

		for word in words:
			if (word.decode('utf8')==word):
				subword =nltk.word_tokenize(word.decode('utf8'))
				for w in subword:
					print w
					freq[w] = freq.get(w, 0) + 1
			else :
				print word
				freq[word] = freq.get(word, 0) + 1
			#print"s", subword 

		#print words
"""			
		#if(word=='every'):
		#	c=c+1
		
	#print c  

	# Sort the frequencies
	wordCounts = reduce(lambda x, y: x + y, freq.values())
	freqSort = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
	# Prune the vocab
	freqSort = freqSort[:pruneThreshold]
	prunedWordCounts = reduce(lambda x, y: x + y, [x[1] for x in freqSort])
	vocab = defaultdict()
	rVocab = defaultdict()
	vocab["UNK"] = 0
	rVocab[0] = "UNK"
	vocabID = 0
	for item in freqSort:
		vocabID += 1
		vocab[item[0]] = vocabID
	 	rVocab[vocabID] = item[0]

	return float(prunedWordCounts)/wordCounts, vocab, rVocab

per,vocab,rVocab=parseCorpus('europarl-v7.de-en.en',10000)

with open("data/vocab.mdl", "wb") as m:
		pickle.dump([vocab, rVocab], m)
print "dumped"    

#with open("data/vocab.mdl", "rb") as model:
#    [vocab, rVocab] = pickle.load(model)   
print(len(vocab)) 
print per
#print vocab
