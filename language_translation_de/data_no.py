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
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize

def parseCorpus(iFile):

	with open("data/vocab.mdl", "rb") as model:
		[vocab, rVocab] = pickle.load(model)
		print len(vocab)
	print "loaded"	    

	f = open(iFile, 'rU')
	text=[]
	c=0
	w=0

	for line in f:
		
		
		words = wordpunct_tokenize(line.decode("utf8"))
		data=[]
		for word in words:
			if word in vocab:
				if (vocab[word]<10000):
					data.append(vocab[word])
					c=c+1
			else:
				data.append(0)
				w=w+1
		
		text.append(data)
		
		
	print c ," ",w ," ", (c+w)	
	
	
	return text

text=parseCorpus('europarl-v7.de-en.de')
with open("data/data_no.mdl", "wb") as m:
		pickle.dump(text, m)
print "dumpedd"
#20183962   33548097   53732059




