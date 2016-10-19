'''To prepare data by sorting sentences according to no of words, it will improve the training time'''
import cPickle as pickle

with open("data/data_no.mdl", "rb") as model:
		data_noe= pickle.load(model)
with open("/scratch/intern/language_translation_de/data/data_no.mdl", "rb") as model:
		data_nog= pickle.load(model)     		
with open("data/vocab.mdl", "rb") as model:
		[vocabe, rVocabe] = pickle.load(model)
with open("/scratch/intern/language_translation_de/data/vocab.mdl", "rb") as model:
		[vocabg, rVocabg] = pickle.load(model)		

def len_argsort(seq):
	sorted_index = sorted(range(len(seq)), key=lambda x: len(seq[x][0]+seq[x][1]) )
	data_no_sorted = [seq[i] for i in sorted_index]
	return data_no_sorted
 

data_com=[]
for i in range(len(data_noe)):
	data_com.append((data_noe[i],data_nog[i]))

data_com_sorted=len_argsort(data_com)	

data_noe_sorted=[]
data_nog_sorted=[]

for i in range(len(data_com_sorted)):
	data_noe_sorted.append(data_com_sorted[i][0])
	data_nog_sorted.append(data_com_sorted[i][1])


with open("data/data_noe.mdl", "wb") as m:
		pickle.dump(data_noe_sorted, m)
with open("/scratch/intern/language_translation_de/data/data_nog.mdl", "wb") as m:
		pickle.dump(data_nog_sorted, m)
print "dumped"

with open("data/data_noe1.mdl", "wb") as m:
		pickle.dump(data_noe_sorted[400000:405000], m)
with open("/scratch/intern/language_translation_de/data/data_nog1.mdl", "wb") as m:
		pickle.dump(data_nog_sorted[400000:405000], m)
print "dumped"

	
print (len(data_noe_sorted)) ,"eng"
print (len(data_nog_sorted)) ,"ger"



for i in xrange (0,20000):
	print "i" ,i
	l=[]
	for j in range (len(data_noe_sorted[i])):
		l.append(rVocabe[data_noe_sorted[i][j]])
	print ' '.join(word.encode("utf8") for word in l)

	l=[]
	for j in range (len(data_nog_sorted[i])):
		l.append(rVocabg[data_nog_sorted[i][j]])
	print ' '.join(word.encode("utf8") for word in l)
	

	#print data_noe_sorted[i] , data_nog_sorted[i]

#for i in range (len(data_noe_sorted)):
#	print len(data_noe[i]) ,len(data_nog[i])	


