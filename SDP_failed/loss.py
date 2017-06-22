
from util import *
from nn import *
import numpy as np


# softmax and cross-entropy loss
class softmax_loss_layer(object):

	def __init__(self, label_list, h_dim):
		self.h_dim = h_dim
		self.pred_dim = len(label_list)

		self.label_list = label_list

		self.W = random_vector(-0.1, 0.1, self.pred_dim, h_dim)
		self.b = random_vector(-0.1, 0.1, self.pred_dim)
		# derivative 
		self.d_W = np.zeros((self.pred_dim, h_dim))
		self.d_b = np.zeros(self.pred_dim)

	def update(self, learning_rate, div_num):

		self.W -= learning_rate * self.d_W / div_num
		self.b -= learning_rate * self.d_b / div_num

		# reset d
		self.d_W = np.zeros_like(self.W)
		self.d_b = np.zeros_like(self.b)

	def softmax_prob(self, h):
		pred_prob  = softmax(np.dot(self.W, h) + self.b)
		pred_label = self.label_list[np.argmax(pred_prob)]
		return pred_label, pred_prob 

	def one_hot(self, label):
		vector = np.zeros(self.pred_dim)
		vector[self.label_list.index(label)] = 1.
		return vector

	def loss(self, h, label):
		pred_lebel, pred_prob = self.softmax_prob(h)
		return -np.sum(self.one_hot(label) * np.log(pred_prob))

	def diff(self, h, label):

		pred_label, pred_prob = self.softmax_prob(h)
		# o = W * h + b
		d_o = pred_prob - self.one_hot(label)

		self.d_W += np.outer(d_o, h)
		self.d_b += d_o
		
		# return derivative of h
		return np.dot(self.W.T, d_o)

	def full_step(self, h, label=None):
		# predicted probability
		pred_label, pred_prob = self.softmax_prob(h)
		
		if label is None:
			log_loss = 0
			g_h = 0
		else:
			# log loss 
			log_loss = self.loss(h, label)
			# diff h
			d_h = self.diff(h, label)

		return pred_label, pred_prob, log_loss, d_h
		


def main():
	np.random.seed(10)
	h_dim = 10
	pred_dim = 4
	label = [0,0,1,0]
	h = np.array([0,1,2,3,4,5,6,7,8,9])

	layer = softmax_loss_layer(pred_dim, h_dim)

	for it in range(2):
		print("iter ",it)
		prob, loss, _ = layer.full_step(h, label)
		print(prob)

if __name__ == "__main__":
	main()