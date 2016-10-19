'''utility methoda for data processing'''
import cPickle as pickle
import pprint
import itertools
import numpy as np


with open("data_no_set.mdl", "rb") as model:
		data_no= pickle.load(model)   
print "loaded data for ngram model"
"""
with open("data_no_set.mdl", "wb") as m:
		pickle.dump(data_no[:1000], m)
print "dumped"
"""
def chunksx(l, n):
	"""Yield successive n-sized chunks from l."""
	datax=[]
	for i in xrange(len(l)):
		if ((i+n)<=len(l)):  
			datax.append(l[i:i+n])
		elif((len(l)-i)==2):
			datax.append(l[i:len(l)]+[0])
		elif((len(l)-i)==1):
			datax.append(l[i:len(l)]+[0,0])	
	return datax		 


def chunksy(l, n):
	datay=[]
	datay=l[3:len(l)]+[l[-1],l[-1],-1]
	return datay	 
pprint.pprint(chunksx(range(5),3)+chunksx(range(11),3))
pprint.pprint(chunksy(range(5),3)+chunksy(range(11),3))

#pprint.pprint(list(chunksy(range(100),3))+list(chunksy(range(100),3)))


data_no_chunk_x=[]
data_no_chunk_y=[]
for i in xrange(len(data_no)):

    if(len(data_no[i])>=3):
		 data_no_chunk_x=data_no_chunk_x+chunksx(data_no[i],3)
		 print i , data_no[i]
		 print data_no_chunk_x[-1]
		 data_no_chunk_y=data_no_chunk_y+chunksy(data_no[i],3)
		 print data_no_chunk_y[-1]
"""
for i in range(len(data_no_chunk_x)):
	print"x " ,(data_no_chunk_x[i])
	print"y " ,(data_no_chunk_y[i])
"""		 
