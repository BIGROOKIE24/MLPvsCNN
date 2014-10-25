"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
from inputParser import get_parser
import cPickle as pickle
import gzip
import os
import sys
import time
import traceback
import numpy
from dA import dA
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

#### rectified linear unit
def ReLU(x):
	y = T.maximum(0.0, x)
#	y = T.log(1 + T.exp(x))
	return(y)
#### sigmoid
def Sigmoid(x):
	y = T.nnet.sigmoid(x)
	return(y)
#### tanh
def Tanh(x):
	y = T.tanh(x)
	return(y)
act = {'ReLU' : ReLU,'Sigmoid' : Sigmoid,'Tanh' : Tanh}

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation = Tanh):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height,filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows,#cols)
		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input
		self.activation = activation

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = numpy.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#	pooling size
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))
		# initialize weights with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(numpy.asarray(
			rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
							   borrow=True)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(input=input, filters=self.W,
				filter_shape=filter_shape, image_shape=image_shape)

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(input=conv_out,
											ds=poolsize, ignore_border=True)

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = self.activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]


def evaluate_lenet5(init_learning_rate=0.1 , lr_decay = 1, n_epochs=200,dataset='mnist.pkl.gz',nkerns0=20, nkerns1=50, batch_size=500,pool_size = 2, filtering=5 ,hidden_size=500,height=28,width=28,dW=numpy.zeros((1,1)), dbias=1, denshape=1, hidden_layer = False, conv_layer = False, activation = Tanh, channel = 1):
	""" Demonstrates lenet on MNIST dataset

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: path to the dataset used for training /testing (MNIST here)

	:type nkerns: list of ints
	:param nkerns: number of kernels on each layer
	"""
	nkerns = [nkerns0,nkerns1]
	poolsize = (pool_size,pool_size)
	n_in = hidden_size
	n_out = hidden_size
	rng = numpy.random.RandomState(23455)

	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size

	denW = theano.shared(value=dW,name='denW', borrow=True)
	denB = theano.shared(value=dbias,name='denb', borrow=True)
	# allocate symbolic variables for the data
	index = T.lscalar()	 # index to a [mini]batch
	x = T.matrix('x')	# the data is presented as rasterized images
	y = T.ivector('y')	# the labels are presented as 1D vector of
						# [int] labels
	learning_rate = theano.shared(numpy.asarray(init_learning_rate,
        dtype=theano.config.floatX))
	ishape = (height, width)  # this is the size of MNIST images

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# Reshape matrix of rasterized images of shape (batch_size,28*28)
	# to a 4D tensor, compatible with our LeNetConvPoolLayer
	layer0_input = x.reshape((batch_size, channel, height, width))

	# Construct the first convolutional pooling layer:
	# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
	# maxpooling reduces this further to (24/2,24/2) = (12,12)
	# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
	layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, channel, height, width),filter_shape=(nkerns[0], channel, filtering, filtering), poolsize=poolsize)

	# Construct the second convolutional pooling layer
	# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
	# maxpooling reduces this further to (8/2,8/2) = (4,4)
	# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
	params = []

	l1h = (height-filtering+1)/pool_size
	l1w = (width-filtering+1)/pool_size

	layer1 = LeNetConvPoolLayer(rng, input=layer0.output,image_shape=(batch_size, nkerns[0], l1h, l1w), filter_shape=(nkerns[1], nkerns[0], filtering, filtering), poolsize=poolsize)

	if conv_layer == True:
		l1h = (l1h-filtering+1)/pool_size
		l1w = (l1w-filtering+1)/pool_size
		layer15 = LeNetConvPoolLayer(rng, input=layer1.output,image_shape=(batch_size, nkerns[1], l1h, l1w), filter_shape=(nkerns[1], nkerns[1], filtering, filtering), poolsize=poolsize)
		layer2_input = layer15.output.flatten(2)
		params = layer15.params
	else:
		layer2_input = layer1.output.flatten(2)

	l1h = (l1h-filtering+1)/pool_size
	l1w = (l1w-filtering+1)/pool_size

	layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * (l1h) * (l1w),
						 n_out=n_out, activation= activation)

	prev_layer = layer2.output
	if hidden_layer == True:
		layer25 = HiddenLayer(rng,input=prev_layer, n_in=n_in, n_out=300,activation= activation)
		prev_layer = layer25.output
		n_in = 300
		params += layer25.params

	if denshape != 1:
		layerX = x.reshape((batch_size, height * width * channel))
		unsup_feats = T.nnet.sigmoid(T.dot(layerX, denW) + denB)
		prev_layer = T.concatenate([prev_layer,unsup_feats], axis=1)
		n_in = n_in + denshape

	layer3 = LogisticRegression(input=prev_layer, n_in=n_in, n_out=10)

	cost = layer3.negative_log_likelihood(y)

	test_model = theano.function([index], layer3.errors(y),
			 givens={
				x: test_set_x[index * batch_size: (index + 1) * batch_size],
				y: test_set_y[index * batch_size: (index + 1) * batch_size]})

	validate_model = theano.function([index], layer3.errors(y),
			givens={
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

	params  += layer3.params + layer2.params + layer1.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)

	decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * lr_decay})

	updates = []
	for param_i, grad_i in zip(params, grads):
		updates.append((param_i, param_i - learning_rate * grad_i))

	train_model = theano.function([index], cost, updates=updates,
		  givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%, lrate %f' % \
					  (epoch, minibatch_index + 1, n_train_batches, \
					   this_validation_loss * 100.,learning_rate.get_value(borrow=True)))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)
					print(('	 epoch %i, minibatch %i/%i, test error of best '
						   'model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

			if patience <= iter:
				done_looping = True
				break
		n_learning_rate = decay_learning_rate()
	end_time = time.clock()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i,'\
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

def printParameters(p):
	print p.n_epochs
	print p.batch_size

	print p.learning_rate
	print p.lr_decay

	print p.nkerns0
	print p.nkerns1
	print p.filtering
	print p.pool_size
	print p.hidden_size

	print p.hidden_layer
	print p.conv_layer
	print p.act

	print p.benchmark
	print p.channel
	print p.height
	print p.width




if __name__ == '__main__':
	parser = get_parser()
	p = parser.parse_args()

	if p.da_file is not "":
		pkl_file = open(p.da_file, 'rb')
		denoiser = pickle.load(pkl_file)
		dW = denoiser.W.get_value()
		dbias = denoiser.b.get_value()
		(dummy,denshape) = dW.shape
	else:
		denshape = 1
		dbias = numpy.zeros(1)
		dW = numpy.zeros((1,1))
#	printParameters(p)

	evaluate_lenet5(init_learning_rate = p.learning_rate, lr_decay = p.lr_decay,  n_epochs = p.n_epochs, nkerns0 = p.nkerns0, nkerns1 = p.nkerns1, batch_size = p.batch_size, pool_size = p.pool_size, filtering = p.filtering, hidden_size = p.hidden_size, dW = dW, dbias = dbias, denshape = denshape, dataset = p.benchmark, conv_layer = p.conv_layer, hidden_layer = p.hidden_layer,activation = act[p.act], channel = p.channel, height = p.height, width = p.width)
