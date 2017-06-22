import random
from math import *
from util import *
from loss import *
import numpy as np

# x is a scalar
def sigmoid(x):
	return 1. / (1. + np.exp(-x))

# x is a scalar
def d_sigmoid(x):
	return x * (1. - x)

# x is a scalar
def d_tanh(x):
	return 1. - x ** 2

# x is a vector
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

# x is a vector
def relu(x):
	return np.array(list(map(lambda num: 0 if num <= 0 else num, x)))

def d_relu(x):
	return np.array(list(map(lambda num: 0 if num <= 0 else 1, x)))

# x is a scalar
# None activation function
def nochange(x):
	return x

def d_nochange(x):
	return 1

# random vector values in [a,b) shape args
def random_vector(a, b, *args):
	np.random.seed(0)
	return np.random.rand(*args) * (b-a) + a


###################################### embedding ##################################
###################################################################################

class embedding_unit_pretrained(object):
	
	def __init__(self, path):
		
		self.dic = {}
		self.x_dim = None
		self.embedding_dim = None

		with open(path, 'r', encoding='utf8') as f:
			for line in f:
				line = line.strip().split(' ')
				vector = np.array(list(map(float, line[1:])))
				self.embedding_dim = len(vector)
				self.dic[line[0]] = vector

		self.x_dim = len(self.dic)

	def get_embedding(self, x):
		if x in self.dic:
			return self.dic[x]
		else:
			return np.zeros(self.embedding_dim)

class embedding_param(object):
	def __init__(self, embedding_dim, x_list):
		self.embedding_dim = embedding_dim
		self.x_list = x_list
		self.x_dim = len(x_list)

		# embedding dict
		self.W = random_vector(-0.1, 0.1, embedding_dim, self.x_dim)

		self.d_W = np.zeros((embedding_dim, self.x_dim))

	def update(self, learning_rate, div_num):

		def clip_v(x_v):
			for i in range(len(x_v)):
				x_v[i] = min(max(x_v[i], 1.0), -1.0)
			return x_v

		def clip_m(x_m):
			for x_v in x_m: 
				clip_v(x_v)
			return x_m
		
		self.W -= learning_rate * self.d_W / div_num

		clip_m(self.W)

		# reset
		self.d_W = np.zeros_like(self.W) 

	def one_hot(self, x):
		vector = np.zeros(self.x_dim)
		if x in self.x_list:
			vector[self.x_list.index(x)] = 1.
		return vector

class embedding_unit(object):
	def __init__(self, param, x):
		self.param = param
		self.x = x
		self.one_hot = param.one_hot(x)
		self.embedding = np.dot(param.W, self.one_hot)

	def get_embedding(self):
		return self.embedding

	def backward(self, d_e):
		self.param.d_W += np.outer(d_e, self.one_hot)


################################### nn #################################
########################################################################

class full_connected_param(object):
	def __init__(self, h_dim, x_dim):
		self.h_dim = h_dim
		self.x_dim = x_dim

		self.W = random_vector(-0.1, 0.1, h_dim, x_dim)
		self.b = random_vector(-0.1, 0.1, h_dim)

		self.d_W = np.zeros((h_dim, x_dim))
		self.d_b = np.zeros(h_dim)

	def update(self, learning_rate, div_num, clip=False):

		def clip_v(x_v):
			for i in range(len(x_v)):
				x_v[i] = min(max(x_v[i], 1.0), -1.0)
			return x_v

		def clip_m(x_m):
			for x_v in x_m: 
				clip_v(x_v)
			return x_m

		self.W -= learning_rate * self.d_W / div_num
		self.b -= learning_rate * self.d_b / div_num

		if clip:
			clip_m(self.W)
			clip_v(self.b)

		# reset
		self.d_W = np.zeros_like(self.W)
		self.d_b = np.zeros_like(self.b)


class full_connected_layer(object):
	def __init__(self, param, activation):
		self.param = param

		# activation function
		if activation == sigmoid:
			self.activation = sigmoid
			self.d_activation = d_sigmoid
		elif activation == tanh:
			self.activation = np.tanh
			self.d_activation = d_tanh
		elif activation == relu:
			self.activation = relu
			self.d_activation = d_relu
		else:
			self.activation = nochange
			self.d_activation = d_nochange


	def forward(self, x):
		self.x = x
		self.a = self.activation(np.dot(self.param.W, x) + self.param.b)
		return self.a

	def backward(self, d_a):
		d_input = self.d_activation(self.a) * d_a

		self.param.d_W += np.outer(d_input, self.x)
		self.param.d_b += d_input

		# return d_X
		return np.dot(self.param.W.T, d_input)

	def backward_with_loss(self, y, loss_layer):

		# loss_layer
		pred_prob, loss, d_a = loss_layer.full_step(self.a, y)

		return pred_prob, loss, self.backward_without_loss(d_a)
	



def main():
	np.random.seed(10)

	h_dim = 2
	embedding_dim = 10
	x_dim = 3


	e_param = embedding_param(embedding_dim, [0,1,2,3,4,5,6,7,8,9])
	unit_3 = embedding_unit(e_param, 3)

	nn_param = full_connected_param(h_dim, embedding_dim)

	layer1 = full_connected_layer(nn_param, None)

	softmax_layer = softmax_loss_layer(['a','b','c'], h_dim)


	for cur_iter in range(100):

		print("cur iter: ", cur_iter)

		xs = layer1.forward(unit_3.get_embedding())
		pred_label, pred_prob, loss , d_h = softmax_layer.full_step(xs, 'c')
		print(softmax_layer.one_hot('c'))

		d_x = layer1.backward(d_h)
		print(d_x)

		unit_3.backward(d_x)

		print("y_pred : ", pred_prob)
		print("loss: ", loss)
		nn_param.update(0.1)
		e_param.update(0.1)


if __name__ == "__main__":
	main()
