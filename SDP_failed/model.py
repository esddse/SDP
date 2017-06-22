
import pickle
import copy
import numpy as np
from util import *
from loss import *
from nn import *
from lstm import *

class config(object):

	def __init__(self):
		# resources
		self.word_list      = read_words()
		self.lemma_list     = read_lemmas()
		self.postag_list    = read_postags()
		self.relation_list  = read_relations()

		self.operation_list = read_operations()
		self.operation_list_no_shift  = copy.deepcopy(self.operation_list)
		self.operation_list_no_shift.remove('SHIFT')
		self.operation_list_no_reduce = copy.deepcopy(self.operation_list)
		self.operation_list_no_reduce.remove('REDUCE')

		self.pretrained_word_path = './model/embedding/useful_embedding'

		self.word_num      = len(self.word_list)
		self.lemma_num     = len(self.lemma_list)
		self.postag_num    = len(self.postag_list)
		self.operation_num = len(self.operation_list)
		self.relation_num  = len(self.relation_list)

		# model parameters
		self.embedding_dim           = 50
		self.word_embedding_dim      = 300
		self.lemma_embedding_dim     = 50
		self.relation_embedding_dim  = 24
		self.postag_embedding_dim    = 24
		self.operation_embedding_dim = 16
		self.hidden_dim    = 50

		# training parameters
		self.learning_rate = 0.3


class integrated_embedding_unit(object):
	def __init__(self, config, x, word_embedding_unit, lemma_embedidng_unit, postag_embedding_unit, integrated_param):
		self.config = config
		self.x = x
		
		self.word_embedding_unit   = word_embedding_unit
		self.lemma_embedidng_unit  = lemma_embedidng_unit
		self.postag_embedding_unit = postag_embedding_unit

		self.integrated_layer = full_connected_layer(integrated_param, relu)
		self.embedding = None

	def get_embedding(self):
		if self.embedding is not None:
			return self.embedding
		else:
			word_embedding   = self.word_embedding_unit.get_embedding(self.x['form'])
			lemma_embedding  = self.lemma_embedidng_unit.get_embedding()
			postag_embedding = self.postag_embedding_unit.get_embedding()
			self.embedding   = self.integrated_layer.forward(np.concatenate((
															 word_embedding,
															 lemma_embedding,
															 postag_embedding)))

		return self.embedding

	def backward(self, d_a):
		d_x      = self.integrated_layer.backward(d_a)

		d_lemma  = d_x[self.config.word_embedding_dim : self.config.word_embedding_dim+self.config.lemma_embedding_dim]
		d_postag = d_x[self.config.word_embedding_dim+self.config.lemma_embedding_dim:]

		self.lemma_embedidng_unit.backward(d_lemma)
		self.postag_embedding_unit.backward(d_postag)


class neural_composition_tree(object):
	# head ----relation-----> dependent
	def __init__(self, config, head, dependent=None, relation=None, composition_param=None):
		self.config    = config

		self.head      = head
		self.dependent = dependent
		self.relation  = relation

		self.embedding = None

		if relation == None:
			self.composited_embedding = head.embedding
		else:
			self.composition_layer = full_connected_layer(composition_param, tanh)

	def get_embedding(self):

		# embedding already generated
		if self.embedding is not None:
			return self.embedding

		# compute embedding recursively
		if self.relation == None:
			self.embedding = self.head.get_embedding()
		else:
			head_embedding      = self.head.get_embedding()
			dependent_embedding = self.dependent.get_embedding()
			relation_embedding  = self.relation.get_embedding() 
			self.embedding      = self.composition_layer.forward(np.concatenate((
				                                                 head_embedding,
				                                                 dependent_embedding,
				                                                 relation_embedding))) 

		return self.embedding


	# backward recursively
	def backward(self, d_a):
		if self.relation == None:
			self.head.backward(d_a)
		else:
			d_x         = self.composition_layer.backward(d_a)
			d_head      = d_x[:self.config.embedding_dim]
			d_dependent = d_x[self.config.embedding_dim:2*self.config.embedding_dim]
			d_relation  = d_X[2*self.config.embedding_dim:]

			self.head.backward(d_head)
			self.dependent.backward(d_dependent)
			self.relation.backward(d_relation)


class stack_lstm_item(object):
	def __init__(self, nct, word, index):
		self.nct   = nct
		self.word  = word
		self.index = index



class stack_lstm_parser(object):
	def __init__(self, config):

		self.config = config

		# embedding layer parameters
		self.word_embedding_unit       = embedding_unit_pretrained(config.pretrained_word_path)

		self.lemma_embedding_param     = embedding_param(config.lemma_embedding_dim, config.lemma_list)
		self.postag_embedding_param    = embedding_param(config.postag_embedding_dim, config.postag_list)
		self.operation_embedding_param = embedding_param(config.operation_embedding_dim, config.operation_list)
		self.relation_embedding_param  = embedding_param(config.relation_embedding_dim, config.relation_list)

		self.integrated_embedding_param = full_connected_param(config.embedding_dim, config.word_embedding_dim + config.lemma_embedding_dim + config.postag_embedding_dim)

		# tree composition parameters
		self.composition_param = full_connected_param(config.embedding_dim, 2 * config.word_embedding_dim + config.relation_embedding_dim)

		# 3 stack-lstm
		self.stack_param   = lstm_param(config.hidden_dim, config.embedding_dim)
		self.buffer_param  = lstm_param(config.hidden_dim, config.embedding_dim)
		self.history_param = lstm_param(config.hidden_dim, config.operation_embedding_dim)

		self.stack   = lstm_network(self.stack_param)
		self.buffer  = lstm_network(self.buffer_param)
		self.history = lstm_network(self.history_param)

		self.stack_inputs   = []
		self.buffer_inputs  = []
		self.history_inputs = []

		# MLP softmax classifier
		self.mlp_param_1 = full_connected_param(config.hidden_dim, 3 * config.hidden_dim)
		self.mlp_layer_1 = full_connected_layer(self.mlp_param_1, tanh)
		
		self.mlp_param_2 = full_connected_param(config.hidden_dim, config.hidden_dim)
		self.mlp_layer_2 = full_connected_layer(self.mlp_param_2, tanh)
		
		self.softmax_layer = softmax_loss_layer(config.operation_list, config.hidden_dim)
		self.softmax_layer_no_shift  = softmax_loss_layer(config.operation_list_no_shift, config.hidden_dim)
		self.softmax_layer_no_reduce = softmax_loss_layer(config.operation_list_no_reduce, config.hidden_dim)
 
	def gen_integrated_embeddding_unit(self, x):

		lemma_embedding_unit = embedding_unit(self.lemma_embedding_param, x['lemma'])
		postag_embedding_unit = embedding_unit(self.postag_embedding_param, x['postag'])

		return integrated_embedding_unit(self.config, x, self.word_embedding_unit, lemma_embedding_unit, postag_embedding_unit, self.integrated_embedding_param)



	# -------------------------- basic stack operation --------------------------------------
	# ---------------------------------------------------------------------------------------

	def stack_size(self):
		return len(self.stack_inputs)

	def buffer_size(self):
		return len(self.buffer_inputs)

	def history_size(self):
		return len(self.history_inputs)

	def stack_empty(self):
		return len(self.stack_inputs) == 0

	def buffer_empty(self):
		return len(self.buffer_inputs) == 0

	def top(self, stack_lstm):
		# empty
		if stack_lstm.size() == 0:
			return np.zeros(stack_lstm.param.h_dim)
		else:
			return stack_lstm.cell_list[-1].state.h

	def stack_push(self, item):
		self.stack_inputs.append(item)
		self.stack.forward_one_step(item.nct.get_embedding())

	def buffer_push(self, item):
		self.buffer_inputs.append(item)
		self.buffer.forward_one_step(item.nct.get_embedding())

	def history_push(self, item):
		self.history_inputs.append(item)
		self.history.forward_one_step(item.nct.get_embedding())

	def stack_pop(self):
		self.stack.pop()
		return self.stack_inputs.pop()

	def buffer_pop(self):
		self.buffer.pop()
		return self.buffer_inputs.pop()

	def history_pop(self):
		self.history.pop()
		return self.history_inputs.pop()


	def stack_clear(self):
		self.stack.xs_clear()
		self.stack_inputs = []

	def buffer_clear(self):
		self.buffer.xs_clear()
		self.buffer_inputs = []

	def history_clear(self):
		self.history.xs_clear()
		self.history_inputs = []

	def configuration_clear(self):
		self.stack_clear()
		self.buffer_clear()
		self.history_clear()



	def print_stack(self):
		print('stack: ', end='')
		for item in self.stack_inputs:
			if type(item.word) == str:
				print(item.word,' ',end='')
			else:
				print(item.word['form'],' ',end='')
		print()

	def print_buffer(self):
		print('buffer: ', end='')
		for item in self.buffer_inputs:
			if type(item.word) == str:
				print(item.word,' ',end='')
			else:
				print(item.word['form'],' ',end='')
		print()
	
	def print_history(self):
		print('history: ', end='')
		for item in self.history_inputs:
			print(item.word,' ',end='')
		print()

	def print_configuration(self):
		self.print_stack()
		self.print_buffer()
		self.print_history()

	# ------------------------------- 4 main operations in arc-eager --------------------------
	# -----------------------------------------------------------------------------------------

	def shift(self):
		if self.buffer_empty():
			return
		item = self.buffer_pop()
		self.stack_push(item)

	def reduce(self):
		if self.stack_empty():
			return
		self.stack_pop()

	def root(self):
		self.root_index = self.buffer_inputs[-1].index
		self.reduce()

	# stact_top_item <------- buffer_top_item
	def left_arc(self, relation):
		return 
		left_item  = self.stack_inputs[-1]
		right_item = self.buffer_inputs[-1]

		if left_item.index == -1 or right_item.index == -1:
			return

		self.in_arcs[left_item.index][right_item.index]  = relation
		self.out_arcs[right_item.index][left_item.index] = relation

	# stack_top_item ------> buffer_top_item
	def right_arc(self, relation):
		return 
		left_item  = self.stack_inputs[-1]
		right_item = self.buffer_inputs[-1]

		if left_item.index == -1 or right_item.index == -1:
			return

		self.in_arcs[right_item.index][left_item.index]  = relation
		self.out_arcs[left_item.index][right_item.index] = relation


	# update parameters for all layers
	def update(self, learning_rate, div_num=1):
		# embedding
		self.lemma_embedding_param.update(learning_rate, div_num)
		self.postag_embedding_param.update(learning_rate, div_num)
		self.operation_embedding_param.update(learning_rate, div_num)
		self.relation_embedding_param.update(learning_rate, div_num)

		self.integrated_embedding_param.update(learning_rate, div_num)

		# conposition
		self.composition_param.update(learning_rate, div_num)

		# lstm-stack
		self.stack_param.update(learning_rate, div_num, clip=True)
		self.buffer_param.update(learning_rate, div_num, clip=True)
		self.history_param.update(learning_rate, div_num, clip=True)

		# mlp
		self.mlp_param_1.update(learning_rate, div_num)
		self.mlp_param_2.update(learning_rate, div_num)
		self.softmax_layer.update(learning_rate, div_num)
		self.softmax_layer_no_shift.update(learning_rate, div_num)
		self.softmax_layer_no_reduce.update(learning_rate, div_num)

		# softmax
		self.softmax_layer.update(learning_rate, div_num)

	def full_step(self, xs, ys, train=False):

		self.loss = 0.
		self.root_index = None

		################### init stack ################################
		root_integrate_embedding_unit = self.gen_integrated_embeddding_unit({'form':'ROOT', 'lemma':'ROOT', 'postag':'ROOT'})
		root_nct = neural_composition_tree(self.config, root_integrate_embedding_unit)
		
		self.stack_push(stack_lstm_item(root_nct, 'ROOT', -1))

		################### push sentence into buffer #################
		predictions = []
		self.in_arcs = {}
		self.out_arcs = {}

		xs.reverse()

		for i in range(len(xs)):	
			x_integrated_embedding_unit = self.gen_integrated_embeddding_unit(xs[i])
			x_nct = neural_composition_tree(self.config, x_integrated_embedding_unit)
			self.buffer_push(stack_lstm_item(x_nct, xs[i], i))

			self.in_arcs[i] = {}
			self.out_arcs[i] = {}


		total_acc = 0
		total_arc = 0

		ys_index = 0
		######## generate operation until both stack and buffer empty ############
		# while (not self.stack_empty() or not self.buffer_empty()) and not ys_index == len(ys): 

		while not ys_index == len(ys): 

			mlp_xs = np.concatenate((self.top(self.stack), self.top(self.history), self.top(self.buffer)))
			mlp_hidden_1 = self.mlp_layer_1.forward(mlp_xs)
			mlp_hidden_2 = self.mlp_layer_2.forward(mlp_hidden_1)

			pred_label, pred_prob, loss, d_mlp_layer2_output = self.softmax_layer.full_step(mlp_hidden_2, ys[ys_index])

			predictions.append(pred_label)
			self.loss += loss

			if ys[ys_index] != 'SHIFT' and ys[ys_index] != 'REDUCE':
				total_arc += 1

			################ backward ##################
			if pred_label != ys[ys_index]:

				d_mlp_layer1_output = self.mlp_layer_2.backward(d_mlp_layer2_output)
				d_lstms_output      = self.mlp_layer_1.backward(d_mlp_layer1_output)

				d_stack_output   = d_lstms_output[:self.config.hidden_dim]
				d_buffer_output  = d_lstms_output[self.config.hidden_dim:2*self.config.hidden_dim]
				d_history_output = d_lstms_output[2*self.config.hidden_dim:]

				tmp_zeros_vec = [0.] * self.config.hidden_dim

				if self.stack_size() > 0:
					d_stack_outputs = [tmp_zeros_vec] * (self.stack_size()-1)
					d_stack_outputs.append(d_stack_output)
					d_stack_outputs = np.array(d_stack_outputs)

					d_stack_inputs  = self.stack.back_prop_through_time_without_loss(d_stack_outputs)

					for i in range(self.stack_size()):
						self.stack_inputs[i].nct.backward(d_stack_inputs[i])
		
				if self.buffer_size() > 0:
					d_buffer_outputs = [tmp_zeros_vec] * (self.buffer_size()-1)
					d_buffer_outputs.append(d_buffer_output)
					d_buffer_outputs = np.array(d_buffer_outputs)

					d_buffer_inputs  = self.buffer.back_prop_through_time_without_loss(d_buffer_outputs)

					for i in range(self.buffer_size()):
						self.buffer_inputs[i].nct.backward(d_buffer_inputs[i])


				if self.history_size() > 0:
					d_history_outputs = [tmp_zeros_vec] * (self.history_size()-1)
					d_history_outputs.append(d_history_output)
					d_history_outputs = np.array(d_history_outputs)

					d_history_inputs  = self.history.back_prop_through_time_without_loss(d_history_outputs)

					for i in range(self.history_size()):
						self.history_inputs[i].nct.backward(d_history_inputs[i])

				if train:
					self.update(self.config.learning_rate)

			else:
				if ys[ys_index] != 'SHIFT' and ys[ys_index] != 'REDUCE':
					total_acc += 1

			##################### operation ######################
			# add to history
			operation = ys[ys_index]

			operation_embedding_unit = embedding_unit(self.operation_embedding_param, operation)
			operation_nct            = neural_composition_tree(self.config, operation_embedding_unit)
			self.history_push(stack_lstm_item(operation_nct, operation, 0))


			if operation == 'SHIFT':
				if not self.buffer_empty():
					self.shift()
			elif operation == 'REDUCE':
			 	if not self.stack_empty():
			 		self.reduce()
			elif operation == 'ROOT':
			 	self.root()
			else:
			 	operation = operation.split('__')
			 	operation = operation[1]

			 	if operation == 'LEFT':
			 		self.left_arc(operation[0])
			 	else:
			 		self.right_arc(operation[0])
			
			print('########################################################')
			print('ys: ', ys[ys_index])
			print('pred: ', pred_label)
			print('-----------------------')
			self.print_configuration()
			print('########################################################')
			print()
			a = input()

			ys_index += 1

			

		
		self.configuration_clear()

		cnt = 0
		for i in range(min(len(predictions), len(ys))):
			if predictions[i] == ys[i]:
				cnt += 1
		self.accuracy = total_acc / total_arc

		self.avg_loss = self.loss / min(len(predictions), len(ys))

		return self.avg_loss, self.accuracy, self.in_arcs, self.root_index, predictions 
			


def main():
	pass


if __name__ == "__main__":
	main()