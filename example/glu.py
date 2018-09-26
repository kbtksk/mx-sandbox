# -*- coding: utf-8 -*-
# https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
## model_ctx = mx.cpu()
model_ctx = mx.gpu()

num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform (data, label):
	return data.astype(np.float32)/255, label.astype(np.float32)


train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)


### network constants
num_hidden = 256
weight_scale = .01

## Weight, Bias for 1st Layer
W1 = nd.random.normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = nd.random.normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

## Weight, Bias for 2nd layer
W2 = nd.random.normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = nd.random.normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)


## ditto, for output layer
W3 = nd.random.normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = nd.random.normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

params = [W1, b1, W2, b2, W3, b3]

## set gradient
for param in params:
	param.attach_grad()

## Defination of Activation function
def relu(x):
	return nd.maximum(x, nd.zeros_like(x))

## Softmax Function for output
def softmax (x):
	exp = np.exp(x-nd.max(x))
	partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
	return exp / partition

## Softmax cross-entropy loss Function
def cross_entropy (yhat, y):
	return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

def softmax_cross_entropy (yhat, y):
	return - nd.nansum(y * nd.log_softmax(yhat), axis=0, exclude=True)


## define DNN Model
def net(X):
	## 1st hidden layer
	h1_linear = nd.dot(X, W1) + b1
	h1 = relu(h1_linear)

	## 2nd hidden layer
	h2_linear = nd.dot(h1, W2) + b2
	h2 = relu(h2_linear)


	## output layer
	yhat_linear = nd.dot(h2, W3) + b3
	return yhat_linear

## optimizer defination
def SGD (params, lr):
	for param in params:
		param[:] = param - lr * param.grad

## Evaluation Metric
def evaluate_accuracy (data_iterator, net):
	numerator = 0
	denominator = 0
	for i, (data, label) in enumerate(data_iterator):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data)
		predictions = nd.argmax(output, axis=1)
		numerator += nd.sum(predictions == label)
		denominator += data.shape[0]
	return (numerator / denominator).asscalar()

## training loop
epochs = 10
learning_rate = .001
smoothing_constant = .01

for e in range(epochs):
	cumulative_loss = 0
	for i, (data, label) in enumerate(train_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context (model_ctx)
		label_one_hot = nd.one_hot(label, 10)
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label_one_hot)
		loss.backward()
		SGD(params, learning_rate)
		cumulative_loss += nd.sum(loss).asscalar()

	test_accuracy = evaluate_accuracy(test_data, net)
	train_accuracy = evaluate_accuracy(train_data, net)
	print("Epoch {}. Loss: {}, Train: {}, Test: {}".format(e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


## prediction
def model_predict (net, data):
	output = net(data)
	return nd.argmax(output, axis=1)

samples = 10

# sampling random 10 points on test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
	data = data.as_in_context(model_ctx)
	im = nd.transpose(data, (1,0,2,3))
	im = nd.reshape(im, (28, 10*28, 1))
	imtiles = nd.tile(im, (1, 1, 3))

	#plt.imshow(imtiles.asnumpy())
	#plt.show()
	pred=model_predict(net, data.reshape((-1, 784)))
	print("model predictions are:", pred)
	print("true label:", label)
	break



import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
## model_ctx = mx.cpu()
model_ctx = mx.gpu()

num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform (data, label):
	return data.astype(np.float32)/255, label.astype(np.float32)


train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)


### network constants
num_hidden = 256
weight_scale = .01

## Weight, Bias for 1st Layer
W1 = nd.random.normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = nd.random.normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

## Weight, Bias for 2nd layer
W2 = nd.random.normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = nd.random.normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)


## ditto, for output layer
W3 = nd.random.normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = nd.random.normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

params = [W1, b1, W2, b2, W3, b3]

## set gradient
for param in params:
	param.attach_grad()

## Defination of Activation function
def relu(x):
	return nd.maximum(x, nd.zeros_like(x))

## Softmax Function for output
def softmax (x):
	exp = np.exp(x-nd.max(x))
	partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
	return exp / partition

## Softmax cross-entropy loss Function
def cross_entropy (yhat, y):
	return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

def softmax_cross_entropy (yhat, y):
	return - nd.nansum(y * nd.log_softmax(yhat), axis=0, exclude=True)


## define DNN Model
def net(X):
	## 1st hidden layer
	h1_linear = nd.dot(X, W1) + b1
	h1 = relu(h1_linear)

	## 2nd hidden layer
	h2_linear = nd.dot(h1, W2) + b2
	h2 = relu(h2_linear)


	## output layer
	yhat_linear = nd.dot(h2, W3) + b3
	return yhat_linear

## optimizer defination
def SGD (params, lr):
	for param in params:
		param[:] = param - lr * param.grad

## Evaluation Metric
def evaluate_accuracy (data_iterator, net):
	numerator = 0
	denominator = 0
	for i, (data, label) in enumerate(data_iterator):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data)
		predictions = nd.argmax(output, axis=1)
		numerator += nd.sum(predictions == label)
		denominator += data.shape[0]
	return (numerator / denominator).asscalar()

## training loop
epochs = 10
learning_rate = .001
smoothing_constant = .01

for e in range(epochs):
	cumulative_loss = 0
	for i, (data, label) in enumerate(train_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context (model_ctx)
		label_one_hot = nd.one_hot(label, 10)
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label_one_hot)
		loss.backward()
		SGD(params, learning_rate)
		cumulative_loss += nd.sum(loss).asscalar()

	test_accuracy = evaluate_accuracy(test_data, net)
	train_accuracy = evaluate_accuracy(train_data, net)
	print("Epoch {}. Loss: {}, Train: {}, Test: {}".format(e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


## prediction
def model_predict (net, data):
	output = net(data)
	return nd.argmax(output, axis=1)

samples = 10

# sampling random 10 points on test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
	data = data.as_in_context(model_ctx)
	im = nd.transpose(data, (1,0,2,3))
	im = nd.reshape(im, (28, 10*28, 1))
	imtiles = nd.tile(im, (1, 1, 3))

	#plt.imshow(imtiles.asnumpy())
	#plt.show()
	pred=model_predict(net, data.reshape((-1, 784)))
	print("model predictions are:", pred)
	print("true label:", label)
	break



