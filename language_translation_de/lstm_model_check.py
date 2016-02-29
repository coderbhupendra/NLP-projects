'''Build a tweet sentiment analyzer'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

datasets = {'imdb': (imdb.load_test_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 177
numpy.random.seed(SEED)

def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
	"""
	Used to shuffle the dataset at each iteration.
	"""

	idx_list = numpy.arange(n, dtype="int32")

	if shuffle:
		numpy.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start:
									minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if (minibatch_start != n):
		# Make a minibatch out of what is left
		minibatches.append(idx_list[minibatch_start:])

	return zip(range(len(minibatches)), minibatches) 


def get_dataset(name):
	return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
	"""
	When we reload the model. Needed for the GPU stuff.
	"""
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)


def unzip(zipped):
	"""
	When we pickle the model. Needed for the GPU stuff.
	"""
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params


def dropout_layer(state_before, use_noise, trng):
	proj = tensor.switch(use_noise,
						 (state_before *
						  trng.binomial(state_before.shape,
										p=0.5, n=1,
										dtype=state_before.dtype)),
						 state_before * 0.5)
	return proj


def _p(pp, name):
	return '%s_%s' % (pp, name)




def load_params(path, params):
	pp = numpy.load(path)
	for kk, vv in params.iteritems():
		if kk not in pp:
			raise Warning(
				'%s is not in the archive' % kk)
		params[kk] = pp[kk]

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams


def init_params(options):
	"""
	Global (not LSTM) parameter. For the embeding and the classifier.
	"""
	params = OrderedDict()
	# embedding
	randn = numpy.random.rand(options['n_words'],
							  options['dim_proj'])
	params['Wemb'] = (0.01 * randn).astype(config.floatX)
	#what does this line do 
	params = param_init_lstm(options,
											  params,
											  prefix=options['encoder'])
	# classifier
	params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
											options['ydim']).astype(config.floatX)
	params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)


	return params
def ortho_weight(ndim):
	W = numpy.random.randn(ndim, ndim)
	u, s, v = numpy.linalg.svd(W)
	return u.astype(config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
	"""
	Init the LSTM parameter:

	:see: init_params
	"""
	W = numpy.concatenate([ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj'])], axis=1)

	params[_p(prefix, 'W')] = W
	U = numpy.concatenate([ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj']),
						   ortho_weight(options['dim_proj'])], axis=1)
	params[_p(prefix, 'U')] = U
	b = numpy.zeros((4 * options['dim_proj'],))
	params[_p(prefix, 'b')] = b.astype(config.floatX)
	print "W shape" ,(W.shape)
	print "U shape" ,(U.shape)
	print "b shape" ,(b.shape)
	return params    


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
	#from line 267
	#(tparams, emb, options,prefix=options['encoder'],mask=mask)

	nsteps = state_below.shape[0]
	if state_below.ndim == 3:
		n_samples = state_below.shape[1]
		#print "3d"

	else:
		n_samples = 1
		#print "2d"

	assert mask is not None

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n * dim:(n + 1) * dim]
		   # print "33d"
		else:
			#print "2dd"
			return _x[:, n * dim:(n + 1) * dim]


	def _step(m_, x_, h_, c_):
		preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
		preact += x_

		i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
		f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
		o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
		c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

		c = f * c_ + i * c
		#why are we doing 1-m
		c = m_[:, None] * c + (1. - m_)[:, None] * c_

		h = o * tensor.tanh(c)
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h, c

	state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
				   tparams[_p(prefix, 'b')])

	dim_proj = options['dim_proj']

	rval, updates = theano.scan(_step,
								sequences=[mask, state_below],
								outputs_info=[tensor.alloc(numpy_floatX(0.),
														   n_samples,
														   dim_proj),
											  tensor.alloc(numpy_floatX(0.),
														   n_samples,
														   dim_proj)],
								name=_p(prefix, '_layers'),
								n_steps=nsteps)
	#why are we returning the first value only
	return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
#what is this line doing specificaly 







def build_model(tparams, options):
	trng = RandomStreams(SEED)

	# Used for dropout.
	use_noise = theano.shared(numpy_floatX(0.))

	x = tensor.matrix('x', dtype='int64')
	mask = tensor.matrix('mask', dtype=config.floatX)
	
	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
												n_samples,
												options['dim_proj']])
	#param_init_lstm(options, params, prefix='lstm') and lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
	proj = lstm_layer(tparams, emb, options,
											prefix=options['encoder'],
											mask=mask)

	if options['encoder'] == 'lstm':
		#what sre we doing here
		proj = (proj * mask[:, :, None]).sum(axis=0)
		proj = proj / mask.sum(axis=0)[:, None]
	if options['use_dropout']:
		proj = dropout_layer(proj, use_noise, trng)

	pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

	f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

	off = 1e-8
	if pred.dtype == 'float16':
		off = 1e-6

	
	return use_noise, x, mask, f_pred

def pred_error(f_pred, prepare_data, data, valid_index, verbose=False): 
	"""        f_pred, prepare_data, train, kf
	Just compute the error
	f_pred: Theano fct computing the prediction
	prepare_data: usual prepare_data for that dataset.
	"""
	
	
	x, mask, y = prepare_data([data[0][t] for t in valid_index],
								  numpy.array(data[1])[valid_index],
								  maxlen=None)
	preds = f_pred(x, mask)
		
	return preds


def train_lstm(
	dim_proj=100,  # word embeding dimension and LSTM number of hidden units.
	patience=10,  # Number of epoch to wait before early stop if no progress
	max_epochs=5000,  # The maximum number of epoch to run
	dispFreq=10,  # Display to stdout the training progress every N updates
	decay_c=0.,  # Weight decay for the classifier applied to the U weights.
	
	n_words=10000,  # Vocabulary size
   
	encoder='lstm',  # TODO: can be removed must be lstm.
	
	maxlen=100,  # Sequence longer then this get ignored
	batch_size=60,  # The batch size during training.
	valid_batch_size=64,  # The batch size used for validation/test set.
	dataset='imdb',

	# Parameter for extra option
	noise_std=0.,
	use_dropout=True,  # if False slightly faster, but worst test error
					   # This frequently need a bigger model.
	reload_model=1,  # Path to a saved model we want to start from.
	
):

	# Model options
	model_options = locals().copy()
	print "model options", model_options

	load_test_data, prepare_data = get_dataset(dataset)

	print 'Loading data'
	train= load_test_data(n_words=n_words,
								   maxlen=maxlen)
	
	ydim = numpy.max(train[1]) + 1

	model_options['ydim'] = ydim

	print 'Building model'
	# This create the initial parameters as numpy ndarrays.
	# Dict name (string) -> numpy ndarray
	
	params = init_params(model_options)
	if reload_model:
		load_params('lstm_model.npz', params)

	# This create Theano Shared Variable from the parameters.
	# Dict name (string) -> Theano Tensor Shared Variable
	# params and tparams have different copy of the weights.
	tparams = init_tparams(params)

	# use_noise is for dropout
	(use_noise, x, mask,
	 f_pred) = build_model(tparams, model_options)

	

	

	print "%d train examples" % len(train[0])
	print "%d no of epochid"  % (len(train[0]) / batch_size)

	with open("data/vocab.mdl", "rb") as model:
		[vocab, rVocab] = pkl.load(model)    

	
	
	


	start_time = time.clock()
	
		

			# Get new shuffled index for the training set.
	kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

	c=0
	w=0
	for _, train_index in kf:
			
			use_noise.set_value(1.)
				# Select the random examples for this minibatch
			y = [train[1][t] for t in train_index]
			x = [train[0][t] for t in train_index]

			x_copy=x
			   
			x, mask, y = prepare_data(x, y)
			
			train_err = pred_error(f_pred, prepare_data, train, train_index)
			
			for i in range(len(train_err)):
					

		"""
					l=[]
					for j in range (len(x_copy[i])):
						
									if(x_copy[i][j]>-1):
											l.append(rVocab[x_copy[i][j]]) 
									else: 
											l.append(rVocab[0]) 
										
					print(l)
		"""			print y[i] ," ",train_err[i]

					if (y[i]!=train_err[i]):
						print "wrong"
						w=w+1
					else:
						#print "right"
						c=c+1
	print "c",c
	print "w",w
	print "t",(c+w)
					



			

   


if __name__ == '__main__':
	# See function train for all possible parameter and there definition.
	train_lstm(
		max_epochs=50
		)
	   
	   
