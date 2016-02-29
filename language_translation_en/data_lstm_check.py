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
import copy
def create_data():
	data_no=[]
	data_copy=[]
	with open("data/data_no.mdl", "rb") as model:
		data_copy= pickle.load(model)   
	print "loaded"	
	
	data_no_new_x=[]
	data_no_new_y=[]
	data_no_new=[]
	
	
	for i in range(len(data_copy)):
		line=data_copy[i]
		if(len(line)<=2):
			print "zero"
		else:	
	#		random.shuffle(line)
			data_no_new.append(line)
    

	#data_no_wrong=zip(data_no_new,np.zeros(len(data_no_new),dtype='int'))	
	data_no_wrong=zip(data_no_new,np.ones(len(data_no_new),dtype='int'))	
	

	data_no_total=data_no_wrong
    
	#print data_no_wrong[1],"\n",data_no_correct[1]	

	random.shuffle(data_no_total)	
	data=zip(*data_no_total)
	data_no_new_x=data[0]
	data_no_new_y=data[1]
		
	
	return data_no_new_x,data_no_new_y

data_no_new_x,data_no_new_y=create_data()
#print data_no_new_x
#make train set 
size=len(data_no_new_x)

with open("data/data_no_test_zip2.mdl", "wb") as m:
		pickle.dump([data_no_new_x,data_no_new_y], m)

print "dumped test set "




