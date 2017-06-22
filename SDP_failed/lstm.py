import random
from math import *
from util import *
from loss import *
from nn import *
import numpy as np



class lstm_param(object):
	def __init__(self, h_dim, x_dim):
		self.h_dim = h_dim
		self.x_dim = x_dim
		hx_dim = h_dim + x_dim

		# weight
		self.Wf = random_vector(-0.1, 0.1, h_dim, hx_dim)
		self.Wi = random_vector(-0.1, 0.1, h_dim, hx_dim)
		self.Wc = random_vector(-0.1, 0.1, h_dim, hx_dim)
		self.Wo = random_vector(-0.1, 0.1, h_dim, hx_dim)

		# bias
		self.bf = random_vector(-0.1, 0.1, h_dim)
		self.bi = random_vector(-0.1, 0.1, h_dim)
		self.bc = random_vector(-0.1, 0.1, h_dim)
		self.bo = random_vector(-0.1, 0.1, h_dim)

		# derivative of weight
		self.d_Wf = np.zeros((h_dim, hx_dim))
		self.d_Wi = np.zeros((h_dim, hx_dim))
		self.d_Wc = np.zeros((h_dim, hx_dim))
		self.d_Wo = np.zeros((h_dim, hx_dim))

		# derivative of bias
		self.d_bf = np.zeros(h_dim)
		self.d_bi = np.zeros(h_dim)
		self.d_bc = np.zeros(h_dim)
		self.d_bo = np.zeros(h_dim)

	# add diff to each param
	def update(self, learning_rate, div_num, clip=False):

		def clip_v(x_v):
			for i in range(len(x_v)):
				x_v[i] = min(max(x_v[i], 1.0), -1.0)
			return x_v

		def clip_m(x_m):
			for x_v in x_m: 
				clip_v(x_v)
			return x_m
				

		# weights
		self.Wf -= learning_rate * self.d_Wf / div_num
		self.Wi -= learning_rate * self.d_Wi / div_num
		self.Wc -= learning_rate * self.d_Wc / div_num
		self.Wo -= learning_rate * self.d_Wo / div_num

		# bias
		self.bf -= learning_rate * self.d_bf / div_num
		self.bi -= learning_rate * self.d_bi / div_num
		self.bc -= learning_rate * self.d_bc / div_num
		self.bo -= learning_rate * self.d_bo / div_num

		if clip:
			clip_m(self.Wf)
			clip_m(self.Wi)
			clip_m(self.Wc)
			clip_m(self.Wo)

			clip_v(self.bf)
			clip_v(self.bi)
			clip_v(self.bc)
			clip_v(self.bo)

		# reset diff
		self.d_Wf = np.zeros_like(self.Wf)
		self.d_Wi = np.zeros_like(self.Wi)
		self.d_Wc = np.zeros_like(self.Wc)
		self.d_Wo = np.zeros_like(self.Wo)
		self.d_bf = np.zeros_like(self.bf)
		self.d_bi = np.zeros_like(self.bi)
		self.d_bc = np.zeros_like(self.bc)
		self.d_bo = np.zeros_like(self.bo)

class lstm_state(object):
	def __init__(self, h_dim, x_dim):
		self.f = np.zeros(h_dim)
		self.i = np.zeros(h_dim)
		self.c = np.zeros(h_dim)
		self.C = np.zeros(h_dim)
		self.o = np.zeros(h_dim)
		self.h = np.zeros(h_dim)
		self.bottom_d_h = np.zeros(h_dim)
		self.bottom_d_C = np.zeros(h_dim)

		self.hx = None

class lstm_cell(object):
	def __init__(self, param, state):
		self.state = state
		self.param = param
		self.hx = None

	def forward(self, x, C_prev=None, h_prev=None):
		# first cell
		if C_prev is None: C_prev = np.zeros_like(self.state.C)
		if h_prev is None: h_prev = np.zeros_like(self.state.h)
		self.C_prev = C_prev
		self.h_prev = h_prev

		# forward pass
		hx = np.concatenate((h_prev, x), axis=0)
		self.state.f = sigmoid(np.dot(self.param.Wf, hx) + self.param.bf)
		self.state.i = sigmoid(np.dot(self.param.Wi, hx) + self.param.bi)
		self.state.c = np.tanh(np.dot(self.param.Wc, hx) + self.param.bc)
		self.state.C = self.state.f * C_prev + self.state.i * self.state.c
		self.state.o = sigmoid(np.dot(self.param.Wo, hx) + self.param.bo)
		self.state.h = self.state.o * self.state.C

		# save hx for backprop
		self.hx = hx

		# output
		return self.state.h

	def backward(self, d_h, d_C):

		dC = d_h * self.state.o + d_C
		do = d_h * self.state.C
		di = self.state.c * dC
		dc = self.state.i * dC
		df = self.C_prev * dC

		# diff in sigmoid / tanh function
		di_input = d_sigmoid(self.state.i) * di
		df_input = d_sigmoid(self.state.f) * df
		do_input = d_sigmoid(self.state.o) * do
		dc_input = d_tanh(self.state.c) * dc

		# diffs in W
		self.param.d_Wi += np.outer(di_input, self.hx)
		self.param.d_Wf += np.outer(df_input, self.hx)
		self.param.d_Wo += np.outer(do_input, self.hx)
		self.param.d_Wc += np.outer(dc_input, self.hx)

		# diffs in b
		self.param.d_bi += di_input
		self.param.d_bf += df_input
		self.param.d_bo += do_input
		self.param.d_bc += dc_input

		# dhx
		dhx = np.zeros_like(self.hx)
		dhx += np.dot(self.param.Wi.T, di_input)
		dhx += np.dot(self.param.Wf.T, df_input)
		dhx += np.dot(self.param.Wo.T, do_input)
		dhx += np.dot(self.param.Wc.T, dc_input)

		# save for the prev cell
		self.state.bottom_d_C = dC * self.state.f
		self.state.bottom_d_h = dhx[:self.param.h_dim]

		# return dx
		return dhx[self.param.h_dim:] 



class lstm_network(object):
	def __init__(self, param):
		self.param = param
		self.cell_list = []
		self.xs = []


	def size(self):
		return len(self.xs)

	def pop(self):
		self.cell_list.pop()
		self.xs.pop()

	def forward_one_step(self, x):
		self.xs.append(x)
		
		# add one cell
		if len(self.xs) > len(self.cell_list):
			state = lstm_state(self.param.h_dim, self.param.x_dim)
			self.cell_list.append(lstm_cell(self.param, state))

		# forward pass
		index = len(self.xs) - 1
		# the first cell
		if index == 0:
			self.cell_list[index].forward(x)
		# other cell
		else:
			C_prev = self.cell_list[index-1].state.C
			h_prev = self.cell_list[index-1].state.h
			self.cell_list[index].forward(x, C_prev, h_prev)


	def back_prop_through_time_without_loss(self, d_hs):

		index = len(self.xs) - 1

		d_xs = []
		# the last cell
		d_h = d_hs[index]
		d_C = np.zeros(self.param.h_dim)
		d_xs.append(self.cell_list[index].backward(d_h, d_C))

		index -= 1

		# backprop through time
		while index >= 0:
			d_h = self.cell_list[index + 1].state.bottom_d_h + d_hs[index]
			d_C = self.cell_list[index + 1].state.bottom_d_C
			d_xs.append(self.cell_list[index].backward(d_h, d_C))
			index -= 1

		d_xs.reverse()

		# print(self.size(), ' ', len(d_xs))

		return d_xs

	def back_prop_through_time_with_loss(self, ys, loss_layer):

		index = len(self.xs) - 1

		d_xs = []
		pred_probs = []
		# the last cell
		pred_prob, loss, d_h = loss_layer.full_step(self.cell_list[index].state.h, ys[index])
		pred_probs.append(pred_prob)
		d_C = np.zeros(self.param.h_dim)
		d_xs.append(self.cell_list[index].backward(d_h, d_C))

		index -= 1

		# backprop through time
		while index >= 0:
			pred_prob, loss_cell, d_h = loss_layer.full_step(self.cell_list[index].state.h, ys[index])
			pred_probs.append(pred_prob)
			loss += loss_cell
			d_h += self.cell_list[index + 1].state.bottom_d_h
			d_C = self.cell_list[index + 1].state.bottom_d_C
			d_xs.append(self.cell_list[index].backward(d_h, d_C))
			index -= 1

		pred_probs.reverse()
		d_xs.reverse()

		return pred_probs, loss, d_xs


	def xs_clear(self):
		self.xs = []
		self.cell_list = []



def main():
	np.random.seed(10)

	h_dim = 100
	x_dim = 50


	param = lstm_param(h_dim, x_dim)
	lstm = lstm_network(param)
	softmax_layer = softmax_loss_layer(5, h_dim) 

	y = [[0,0,1,0,0], [1,0,0,0,0], [0,0,0,1,0]]

	

	for cur_iter in range(1000):
		times = 1
		times = random.randint(1,10)
		y_list = y * times
		input_var_arr = [np.random.random(x_dim) for _ in y_list]
		print("cur iter: " + str(cur_iter))

		for ind in range(len(y_list)):
			lstm.forward_one_step(input_var_arr[ind])
			

		pred_probs, loss, _ = lstm.back_prop_through_time_with_loss(y_list, softmax_layer)
		print(list(map(np.argmax, pred_probs)))
		print("loss: ", str(loss))
		param.update(learning_rate=0.1)
		lstm.xs_clear()


if __name__ == "__main__":
	main()