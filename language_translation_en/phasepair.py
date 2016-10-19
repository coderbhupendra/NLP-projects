'''To prepare dataset pairwise for English and its corresponding German Sentence'''
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

def getPhrasePairs():
	"""
	
	Returns:
		phrasePairs : Tuples containing (source phrase vector, target phrase vector, target phrase ids)
	
	"""
	sEmbeddings = np.load('data_LM/Wembe.npy')
	tEmbeddings = np.load('data_LM/Wembg.npy')
	with open("data/data_no.mdl", "rb") as model:
		data_noe= pickle.load(model)   
	with open("/scratch/intern/language_translation_en/data/data_no.mdl", "rb") as model:
		data_nog= pickle.load(model)     
	print "loaded data_no's"
	phrasePairs = []
	for i in xrange(len(data_nog)):
		print i
		if ((not 10000 in data_nog[i])or(not 10000 in data_noe[i]))  :

			sPhrase = sEmbeddings[data_noe[i]]
			tPhrase = tEmbeddings[data_nog[i]]

		# Don't include phrases that contain only OOVs
		if np.sum(sPhrase) == 0 or np.sum(tPhrase) == 0:
			continue
		phrasePairs.append((sPhrase,tPhrase,data_nog[i]))
	print "made phrasePairs"
	return phrasePairs

phrasePairs=getPhrasePairs()

with open("data/phrasePairs.mdl", "wb") as m:
	pickle.dump(phrasePairs, m)
print "dumpedd"
