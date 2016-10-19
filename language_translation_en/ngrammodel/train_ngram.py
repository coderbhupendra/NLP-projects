'''Train N-gram model and save word embedding for vocabulary'''
import theano
import numpy
import os
import cPickle as pickle
import time
theano.config.floatX='float32'
from theano import tensor as T
from collections import OrderedDict

#theano.config.exception_verbosity = 'high'
#theano.config.optimizer = 'None'
#theano.allow_gc = False
#theano.traceback_limit = -1
#theano.profile = True
#theano.profile_optimizer = True
#theano.profile_memory = True


class model(object):
    
    def __init__(self, nh, nc, de, cs,save):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''

        if not save:
        # parameters of the model
            self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nc, de)).astype(theano.config.floatX)) # add one for PADDING at the end
            self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de*cs, nh)).astype(theano.config.floatX))
            self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
            self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
            self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
            self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
            self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        else:
            self.emb = theano.shared(value=numpy.load("model/embeddings.npy"),name='emb',borrow=True)
            self.Wh=theano.shared(value=numpy.load("model/Wh.npy"),name='Wh',borrow=True)
            self.Wx=theano.shared(value=numpy.load("model/Wx.npy"),name='Wx',borrow=True)
            self.W=theano.shared(value=numpy.load("model/W.npy"),name='W',borrow=True)
            self.bh=theano.shared(value=numpy.load("model/bh.npy"),name='bh',borrow=True)
            self.b=theano.shared(value=numpy.load("model/b.npy"),name='b',borrow=True)
            self.h0=theano.shared(value=numpy.load("model/h0.npy"),name='h0',borrow=True)
            
        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0],de*cs))
        y = T.ivector('y') # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b).flatten()
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x=s
        #p_y_given_x_sentence = s[:,:] 
        #y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        rows=T.arange(p_y_given_x.shape[0])
        prob=p_y_given_x[rows, y]
        nll = -T.mean(T.log(prob))
        #nll = -T.mean(T.log(p_y_given_x)[y])

        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        #self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        
    def save(self, folder): 
        print "saving.."  
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

        
    
    
if __name__ == '__main__':
    rnn = model(nh = 100,nc = 10000,de = 100,cs = 3,save=1)
    with open("/scratch/intern/language_translation_en/data/data_noe.mdl", "rb") as model:
            data_no= pickle.load(model)   
    print "loaded data for ngram model"

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

    
for e in xrange(3,15):
        cost=0
        tic =time.time()
        print len(data_no)
        for i in xrange(len(data_no)):
            if(len(data_no[i])>=3):
                #print data_no[i]
                #print chunksx(data_no[i],3) ,chunksy(data_no[i],3)
                x=chunksx(data_no[i],3)
                y=chunksy(data_no[i],3)
                cost=cost+(rnn.train(x,y,.08))
                #print i,len(data_no[i]), cost ,y
                
                    

            if (((i%1000)==0)& (i>0)):

                #print rnn.W[3][1].eval() ,rnn.emb[3][1].eval()
                rnn.save("model")
                print i ,e ,"time:",(time.time()-tic) ,"cost ," ,(cost/1000)
                cost=0
                tic =time.time()


        
