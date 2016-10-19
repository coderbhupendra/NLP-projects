'''To calulate the closeness fo two word embedding using euclidean distnace''' 
import numpy as np
import cPickle as pickle
#change the vocabs
with open("data/vocab.mdl", "rb") as model:
		[vocabe, rVocabe] = pickle.load(model)
with open("/home/bhupendra/deeplearning//scratch/intern/language_translation_de/data/vocab.mdl", "rb") as model:
		[vocabg, rVocabg] = pickle.load(model)      


#sdata = np.load('lstm_model.npz')
#sEmbeddings=sdata['Wemb.npy']
sEmbeddings=np.load('/home/bhupendra/deeplearning/scratch/intern/language_translation_en/ngrammodel/model/embeddings.npy')


#tdata = np.load('/scratch/intern/language_translation_de/lstm_model.npz')
#tEmbeddings=tdata['Wemb.npy']
tEmbeddings=np.load('/home/bhupendra/deeplearning/scratch/intern/language_translation_de/ngrammodel/model/embeddings.npy')

sEmbeddings=np.array(sEmbeddings,dtype='float32')
tEmbeddings=np.array(tEmbeddings,dtype='float32')

import random
import scipy
from scipy import spatial

my_randoms = random.sample(xrange(10000), 10)

for l in range(len(my_randoms)):
	idx=my_randoms[l]
	print "key word ",rVocabe[idx] , idx
	index=[]
	for i in range(len(sEmbeddings)):
		x=np.array( [ sEmbeddings[idx], sEmbeddings[i] ] )
		cost = float(scipy.spatial.distance.pdist(x))
		index.append((cost,i))
	sorted_by_cost= sorted(index, key=lambda tup: tup[0])

	li=sorted_by_cost[0:10]	

	for k in range(len(li)):
		print rVocabe[li[k][1]] , li[k][1]
	print "\n \n"	
