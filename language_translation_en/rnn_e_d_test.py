#!/usr/bin/env python
import sys
import time
import os
import numpy
import theano
import theano.typed_list
from theano import tensor as T
from collections import OrderedDict
theano.config.warn_float64='warn'
theano.config.floatX='float32'
#### Activate the commented section that follows for debugging ####
#theano.config.exception_verbosity = 'high'
#theano.config.optimizer = 'None'
#theano.allow_gc = False
#theano.traceback_limit = -1
#theano.profile = True
#theano.profile_optimizer = True
#theano.profile_memory = True


class RNNED(object):
  """ RNN encoder-decoder """

  def __init__(self, nh, nc, de, model=None):
    """
    Parameters:
    Hyperparameters used for initialization
    nh : dimension of the hidden layers
    nc : number of classes (labels)
    de : dimension of embedding
    model : A list of pre-trained parameters
    """
    # The hidden layer at time t=0
    self.h0 = theano.shared(name='h0',
        value = numpy.zeros(nh,
          dtype=theano.config.floatX))

    # For the decoder, to combine at time 0, this represents the input at time -1
    self.y0 = theano.shared(name='y0',
        value = numpy.zeros(de,
          dtype=theano.config.floatX))

    self.dim=de    

    if model is not None:
      # Pre-trained parameters supplied
      print "*** Loading pre-trained model"
      [self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.V_e, \
        self.V_d, self.W_d, self.U_d, self.C_d, self.W_z_d, self.U_z_d, self.C_z_d, self.W_r_d, \
        self.U_r_d, self.C_r_d, self.O_h, self.O_y, self.O_c, self.G] = [theano.shared(p) for p in model]

    else:
      # Parameter : Weight matrix for the encoder hidden state
      self.W_e = theano.shared(name='W_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the encoder hidden state
      self.U_e = theano.shared(name='U_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the encoder hidden state : Reset gate
      self.W_z_e = theano.shared(name='W_z_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the encoder hidden state : Reset gate
      self.U_z_e = theano.shared(name='U_z_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the encoder hidden state : Update gate
      self.W_r_e = theano.shared(name='W_r_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the encoder hidden state : Update gate
      self.U_r_e = theano.shared(name='U_r_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the computation of the context vector
      self.V_e = theano.shared(name='V_e',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (de, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the initialization of the hidden state of the decoder
      self.V_d = theano.shared(name='V_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state
      self.W_d = theano.shared(name='W_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state
      self.U_d = theano.shared(name='U_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state
      self.C_d = theano.shared(name='C_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Update gate
      self.W_z_d = theano.shared(name='W_z_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Update gate
      self.U_z_d = theano.shared(name='U_z_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Update gate
      self.C_z_d = theano.shared(name='C_z_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Reset gate
      self.W_r_d = theano.shared(name='W_r_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Reset gate
      self.U_r_d = theano.shared(name='U_r_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                  .astype(theano.config.floatX))

      # Parameter : Weight matrix for the decoder hidden state : Reset gate
      self.C_r_d = theano.shared(name='C_r_d',
                  value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                  .astype(theano.config.floatX))

      self.O_h = theano.shared(name='O_h',
                  value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, nh))
                  .astype(theano.config.floatX))

      self.O_y = theano.shared(name='O_y',
                  value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, de))
                  .astype(theano.config.floatX))

      self.O_c = theano.shared(name='O_c',
                  value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, de))
                  .astype(theano.config.floatX))

      self.G = theano.shared(name='G',
                  value=0.2 * numpy.random.uniform(-1.0, 1.0, (nc, de))
                  .astype(theano.config.floatX))

    # Bundle the parameters
    self.encoderParams = [self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.V_e]
    self.encoderNames = ['W_e', 'U_e', 'W_z_e', 'U_z_e', 'W_r_e', 'U_r_e', 'V_e']
    self.decoderParams = [self.V_d, self.W_d, self.U_d, self.C_d, self.W_z_d, self.U_z_d, self.C_z_d, self.W_r_d, \
        self.U_r_d, self.C_r_d, self.O_h, self.O_y, self.O_c, self.G]
    self.decoderNames = ['V_d', 'W_d', 'U_d', 'C_d', 'W_z_d', 'U_z_d', 'C_z_d', 'W_r_d', 'U_r_d', 'C_r_d', 'O_h', 'O_y', 'O_c', 'G']
    self.params = self.encoderParams + self.decoderParams
    self.names = self.encoderNames + self.decoderNames

    # Accumulates gradients
    self.batch_gradients = [theano.shared(value = numpy.zeros((p.get_value().shape)).astype(theano.config.floatX)) for p in self.params]
    # Dummy shared variable that will be updated to accumulate NLLs
    self.batch_nll = theano.shared(value = numpy.zeros((1)).astype(theano.config.floatX))

    # Compile training function
    self.prepare_train(de)

  def prepare_train(self, de):
    """
    Prepares training for the RNN encoder-decoder model
    Compiles training and testing functions in Theano that can be accessed from an
      external function

    Parameters:
      de : The dimensionality of the word embeddings (input)
    """
    ## Prepare to recieve input and output labels
    X = T.fmatrix('X')
    Y = T.iscalar('Y')


    def encoder(x):
      """
      The encoder of the RNN encoder-decoder model

      Parameters:
        x : The input phrase/sentence : The RNN will apply its recurrence over this sequence

      Returns:
        c : The context vector computed from the final hidden state of the RNN which represents
          the embedding for the entire input sequence
      """

      def encoder_recurrence(x_t, h_tm1):
        """
        The recurrence of the Encoder RNN

        Parameters:
          x_t : The input at time t
          h_tm1 : The hidden state at time t-1

        Returns:
          h_t : The hidden state at time t
        """
        # Reset gate
        r = T.nnet.sigmoid(T.dot(self.W_r_e, x_t) + T.dot(self.U_r_e, h_tm1))
        # Update gate
        z = T.nnet.sigmoid(T.dot(self.W_z_e, x_t) + T.dot(self.U_z_e, h_tm1))
        # Gated output
        h_prime = T.tanh(T.dot(self.W_e, x_t) + T.dot(self.U_e, r * h_tm1))
        # Compute hidden state
        h_t = z * h_tm1 + (1 - z) * h_prime
        return h_t

      # Encoder Recurrence via Theano scan
      h_e, _ = theano.scan(fn=encoder_recurrence,
          sequences = x,
          outputs_info=self.h0,
          n_steps=x.shape[0])

      ## Compute the context vector from the final hidden state
      c = T.tanh(T.dot(self.V_e, h_e[-1]))
      return c


    def decoder(x,y):
      """
      The decoder of the RNN encoder-decoder model
      Invokes the encoder of this model to get the context vector for the input

      Parameters:
        x : The input phrase/sentence : The RNN will apply its recurrence over this sequence
        y : The output phrase/sentence : The decoder will apply its recurrence over this sequence
        y_idx : A vector containg the vocab (target) indices for the current output phrase so that
                they can be easily looked up in the softmax output
        train_flag : (Optional) A flag which enables training (enabled by default). Basically, this function returns
                  gradients only when this flag is set

      Returns:
        phrase_nll : The output of the objective function : The negative log-likelihood of the output
        phrase_grads : (Conditional : Check train_flag) The gradients of the objective function with respect to the parameters of the model
                        These are cached and the parameters are updated only once per batch to improve stability
      """
      # Get the context vector for the input from the encoder
      c = encoder(x)
      # Initialize the hidden state
      h_d_0 = T.tanh(T.dot(self.V_d, c))
      

      def decoder_recurrence(t,h_tm1, y_tm1, c,y):
        """
        The recurrence of the Decoder RNN

        Parameters:
          t : The current time
          h_tm1 : The hidden state (of the decoder) at time (t-1)
          y_idx : A vector containing the vocab indices for the current output phrase
          c : The context vector for the input (x) obtained from the encoder
          y : The output phrase / sentence

        Returns:
          h_t : The hidden state at time t
          nll_t : The negative log-likelihood for the output at time t
                  Note that we can do this becuase this objective function is decomposable
        """
        if t == 0:
          y_tm1 = self.y0

        # Reset gate
        r = T.nnet.sigmoid(T.dot(self.W_r_d, y_tm1) + T.dot(self.U_r_d, h_tm1) + T.dot(self.C_r_d, c))
        # Update gate
        z = T.nnet.sigmoid(T.dot(self.W_z_d, y_tm1) + T.dot(self.U_z_d, h_tm1) + T.dot(self.C_z_d, c))
        # Gated output
        h_prime = T.tanh(T.dot(self.W_d, y_tm1) +  r * (T.dot(self.U_d, r * h_tm1) + T.dot(self.C_d, c)))
        # Compute hidden state
        h_t = z * h_tm1 + (1 - z) * h_prime
        # Compute the final layer
      
        y_t = T.dot(self.O_h, h_t) + T.dot(self.O_y, y_tm1) + T.dot(self.O_c, c)
       

        # Softmax to get probabilities over target vocab
        p_t = T.nnet.softmax(T.dot(self.G, y_t))
        
        # y for this time is observed, return the NLL for this observation
        idy=T.argmax(p_t)
        
        return [h_t, y_t,idy]

      # Apply the decoder recurrence through theano scan
      [h_t, y_t,idy], _ = theano.scan(fn=decoder_recurrence,
          sequences = T.arange(y),
          outputs_info = [h_d_0, self.y0,None],
          non_sequences = [c,y],
          n_steps = y)

      # Compute the average NLL for this phrase
      
        
      return idy

    # Learning rate
    lr = T.scalar('lr')
    # example index
    i = T.iscalar('i')

    # Get the average phrase NLL and the gradients
    translation= decoder(X,Y)
    
    # Compile theano functions for training and testing
    self.phrase_train = theano.function(inputs=[X,Y], outputs=translation ,on_unused_input='warn')


  def train(self, batch):
    """
    Trains the RNN encoder-decoder model for a minibatch
      Invokes compiled theano functions for training wrt each example

    Parameters:
      batch : A minibatch containing tuples of the input(x), output(y) and the outut vocab indices (y_idx)
      lr : The learning rate for SGD
    """
    batch_trans=[]
    # Train wrt each example in the batch
    
    for i, (x,y) in enumerate(batch):
      
      translation=self.phrase_train(x,y)
      batch_trans.append(translation)
      
   
    return batch_trans


  

  def getParams(self):
    """
    Returns the values of the shared parameters (for pickling)
    """
    return [p.get_value() for p in self.params]


  
