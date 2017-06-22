# coding:utf-8

import pickle
import copy 

from util import *

import numpy as np

model_dir = './model'
model_name = 'model_'


class Perceptron(object):
	def __init__(self, tags):
		self.tags = tags
		self.tags_no_arc = ['REDUCE','SHIFT']
		self.weights = {}
		self.acc_weights = {}
		self.last_step = {}
		self.step = 0

	def get_weight(self, feature):
		return self.weights[feature] if feature in self.weights else 0

	def gen_tagged_features(self, features, tag):
		return list(map(lambda feature: tag+feature, features))

	def update_weight(self, feature, delta):
		if feature not in self.weights:
			self.weights[feature] = 0
			self.acc_weights[feature] = 0
			self.last_step[feature] = self.step
		else:
			self.acc_weights[feature] += (self.step - self.last_step[feature]) * self.weights[feature]
			self.last_step[feature] = self.step

		self.weights[feature] += delta

	def update(self, configuration, tag, delta):
		features = configuration.gen_features()
		features = self.gen_tagged_features(features, tag)
		for feature in features:
			self.update_weight(feature, delta)

	def average(self):
		for feature, acc_weights in self.acc_weights.items():
			if self.last_step[feature] != self.step:
				acc_weights += (self.step - self.last_step[feature]) * self.weights[feature]
			self.weights[feature] = acc_weights / self.step

		self.step = 0

	def predict(self, configuration):
		
		if configuration.stack[-1].lemma == '__BOTTOM__':
			return 'SHIFT'
		elif configuration.buffer[-1].lemma == '__BOTTOM__':
			return 'REDUCE'
		

		features = configuration.gen_features()
		scores = []
		
		for tag in self.tags:
			tagged_features = self.gen_tagged_features(features, tag)
			scores.append(sum(list(map(self.get_weight, tagged_features))))

		self.step += 1
		return self.tags[np.argmax(scores)]
'''

def polynomial(features_1, features_2, dim=2):
	same = 0.
	for i in range(len(features_1)):
		if features_1[i] == features_2[i]:
			same += 1
	return (1 + same) ** dim

def RBF(features_1, features_2, sigma=1.):
	diff = 0.
	for i in range(len(features_1)):
		if features_1[i] != features_2[i]:
			diff += 1
	return np.exp(- diff ** 2 / (2 * sigma ** 2))



class Perceptron(object):
	def __init__(self, tags, kernel):
		self.kernel = kernel
		self.tags = tags
		self.tags_no_arc = ['REDUCE','SHIFT']
		self.weights = {}
		self.acc_weights = {}
		self.last_step = {}
		self.step = 0

	def get_weight(self, features):
		return self.weights[features] if features in self.weights else 0

	def gen_tagged_features(self, features, tag):
		return tuple(map(lambda feature: tag+feature, features))

	def update_weight(self, features, delta):
		if features not in self.weights:
			self.weights[features] = 0
			self.acc_weights[features] = 0
			self.last_step[features] = self.step
		else:
			self.acc_weights[features] += (self.step - self.last_step[features]) * self.weights[features]
			self.last_step[features] = self.step

		self.weights[features] += delta

	def update(self, configuration, tag, delta):
		features = configuration.gen_features()
		features = self.gen_tagged_features(features, tag)
		self.update_weight(features, delta)

	def average(self):
		for features, acc_weights in self.acc_weights.items():
			if self.last_step[features] != self.step:
				acc_weights += (self.step - self.last_step[features]) * self.weights[features]
			self.weights[features] = acc_weights / self.step

		self.step = 0

	def predict(self, configuration):
		
		if configuration.stack[-1].lemma == '__BOTTOM__':
			return 'SHIFT'
		elif configuration.buffer[-1].lemma == '__BOTTOM__':
			return 'REDUCE'
		

		features = configuration.gen_features()
		scores = []
		
		for tag in self.tags:
			score = 0
			tagged_features = self.gen_tagged_features(features, tag)
			for stored_features, weight in self.weights.items():
				score += weight * self.kernel(stored_features, tagged_features)
			scores.append(score)

		self.step += 1
		return self.tags[np.argmax(scores)]
'''


def save_model(model, path):
	with open(path, 'wb') as f:
		pickle.dump(model, f)

def load_model(path):
	with open(path, 'rb') as f:
		model = pickle.load(f)
		return model


#
#
# 
#
# 
# item that store some attributes of a word 

class ConfigurationItem(object):
	def __init__(self, form, lemma, postag):
		self.form   = form
		self.lemma  = lemma
		self.postag = postag


# create a hash table for a sentence
def gen_indices(sentence):
	index_2_item = {}
	item_2_index = {}

	bottom_item = ConfigurationItem('__BOTTOM__', '__BOTTOM__', '__BOTTOM__')
	root_item   = ConfigurationItem('__ROOT__', '__ROOT__', '__ROOT__')

	index_2_item[-2] = bottom_item
	item_2_index[bottom_item] = -2
	index_2_item[-1] = root_item
	item_2_index[root_item] = -1

	for index in range(len(sentence)):
		index_2_item[index] = sentence[index]
		item_2_index[sentence[index]] = index

	return index_2_item, item_2_index 

class DependencyGraph(object):

	def __init__(self, index_2_item, item_2_index):
		self.size = len(index_2_item)
		self.index_2_item = index_2_item
		self.item_2_index = item_2_index
		self.in_arc = {}
		self.out_arc = {}

		for index, item in index_2_item.items():
			self.in_arc[index] = []
			self.out_arc[index] = []


	def in_arc_num(self, index):
		return len(self.in_arc[index]) - 2

	def out_arc_num(self, index):
		return len(self.out_arc[index]) - 2

	def left_most_child(self, index):
		return self.out_arc[index][0][0] if len(self.out_arc[index]) != 0 else -2

	def right_most_child(self, index):
		return self.out_arc[index][-1][0] if len(self.out_arc[index]) != 0 else -2

	def left_most_parent(self, index):
		return self.in_arc[index][0][0] if len(self.in_arc[index]) != 0 else -2

	def right_most_parent(self, index):
		return self.in_arc[index][-1][0] if len(self.in_arc[index]) != 0 else -2


	# u <----relation---- v
	def left_arc(self, u_index, v_index, relation):
		if u_index >= self.size or u_index < -2:
			print(u_index, ' not in graph')
			return
		elif v_index >= self.size or v_index < -2:
			print(v_index, ' not in graph')
			return 
		else:
			if (v_index, relation) not in self.in_arc[u_index]:
				self.in_arc[u_index].append((v_index, relation))
				self.out_arc[v_index].insert(0, (u_index, relation))
				return True
			else:
				return False

	# u ----relation------> v
	def right_arc(self, u_index, v_index, relation):
		if u_index > self.size or u_index < -2:
			print(u_index, ' not in graph')
			return
		elif v_index > self.size or v_index < -2:
			print(v_index, ' not in graph')
			return 
		else:
			if (u_index, relation) not in self.in_arc[v_index]:
				self.in_arc[v_index].insert(0, (u_index, relation))
				self.out_arc[u_index].append((v_index,relation))
				return True
			else:
				return False

	def format_output(self, data):
		root = self.out_arc[-1][0] if len(self.out_arc[-1]) != 0 else -1
		preds = []
		for i in range(len(data)):
			# root
			if i == root:
				data[i]['ptop'] = '+'
			else:
				data[i]['ptop'] = '-'
			
			# pred
			if len(self.out_arc[i]) > 0:
				data[i]['ppred'] = '+'
				preds.append(i)
			else:
				data[i]['ppred'] = '-'
		# args
		for i in range(len(data)):
			data[i]['parg'] = ['_'] * len(preds)
			for j in range(len(preds)):
				for item in self.in_arc[i]:
					if item[0] == preds[j]:
						data[i]['parg'][j] = item[1]
	

class Configuration(object):
	def __init__(self, sentence):
		self.sentence = sentence
		self.index_2_item, self.item_2_index = gen_indices(sentence)
		self.stack   = [
			self.index_2_item[-2],
			self.index_2_item[-2],
			self.index_2_item[-1]	
			]
		self.buffer  = []
		self.history = ['__BOTTOM__', '__BOTTOM__', '__BOTTOM__']

		for word in sentence:
			self.buffer.append(word)

		self.buffer.append(self.index_2_item[-2])
		self.buffer.append(self.index_2_item[-2])

		self.buffer.reverse()

		self.arcs = DependencyGraph(self.index_2_item, self.item_2_index)

		self.update_history_ok = False

	def print_stack(self):
		print('stack: ', end='')
		for item in self.stack:
			print(item.lemma, ' ', end='')
		print()

	def print_buffer(self):
		print('buffer: ', end='')
		for item in self.buffer:
			print(item.lemma, ' ', end='')
		print()

	def print_history(self):
		print('history: ', end='')
		for item in self.history:
			print(item, ' ', end='')
		print()

	def print_state(self):
		self.print_stack()
		self.print_buffer()
		self.print_history()
		print()


	def stack_size(self):
		return len(self.stack) - 2

	def buffer_size(self):
		return len(self.buffer) - 2

	def history_size(self):
		return len(self.history) - 2

	def stack_empty(self):
		return self.stack_size() == 0

	def buffer_empty(self):
		return self.buffer_size() == 0

	def history_empty(self):
		return self.history_size() == 0

	def gen_features(self):
		# single word
		single_word_features = (
			's1l'   + self.stack[-1].lemma,
			's1p'   + self.stack[-1].postag,
			's1lp'  + self.stack[-1].lemma + self.stack[-1].postag,
			's2l'   + self.stack[-2].lemma,
			's2p'   + self.stack[-2].postag,
			's2lp'  + self.stack[-2].lemma + self.stack[-2].postag,
			'b1l'   + self.buffer[-1].lemma,
			'b1p'   + self.buffer[-1].postag,
			'b1lp'  + self.buffer[-1].lemma + self.buffer[-1].postag,
			'b2l'   + self.buffer[-2].lemma,
			'b2p'   + self.buffer[-2].postag,
			'b2lp'  + self.buffer[-2].lemma + self.buffer[-2].postag
		)


		# word-pair
		word_pair_features = (
			's1lps2lp'  + self.stack[-1].lemma + self.stack[-1].postag + self.stack[-2].lemma + self.stack[-2].postag,
			's1lps2l'   + self.stack[-1].lemma + self.stack[-1].postag + self.stack[-2].lemma,
			's1lps2p'   + self.stack[-1].lemma + self.stack[-1].postag + self.stack[-2].postag,
			's1ls2lp'   + self.stack[-1].lemma + self.stack[-2].lemma + self.stack[-2].postag,
			's1ps2lp'   + self.stack[-1].postag + self.stack[-2].lemma + self.stack[-2].postag,
			's1ls2l'    + self.stack[-1].lemma + self.stack[-2].lemma,
			's1ps2p'    + self.stack[-1].postag + self.stack[-2].postag,

			's1lpb1lp'  + self.stack[-1].lemma + self.stack[-1].postag + self.buffer[-1].lemma + self.buffer[-1].postag,
			's1lpb1l'   + self.stack[-1].lemma + self.stack[-1].postag + self.buffer[-1].lemma,
			's1lpb1p'   + self.stack[-1].lemma + self.stack[-1].postag + self.buffer[-1].postag,
			's1lb1lp'   + self.stack[-1].lemma + self.buffer[-1].lemma + self.buffer[-1].postag,
			's1pb1lp'   + self.stack[-1].postag + self.buffer[-1].lemma + self.buffer[-1].postag,
			's1lb1l'    + self.stack[-1].lemma + self.buffer[-1].lemma,
			's1pb1p'    + self.stack[-1].postag + self.buffer[-1].postag,

			'b1lpb2lp'  + self.buffer[-1].lemma + self.buffer[-1].postag + self.buffer[-2].lemma + self.buffer[-2].postag,
			'b1lpb2l'   + self.buffer[-1].lemma + self.buffer[-1].postag + self.buffer[-2].lemma,
			'b1lpb2p'   + self.buffer[-1].lemma + self.buffer[-1].postag + self.buffer[-2].postag,
			'b1lb2lp'   + self.buffer[-1].lemma + self.buffer[-2].lemma + self.buffer[-2].postag,
			'b1pb2lp'   + self.buffer[-1].postag + self.buffer[-2].lemma + self.buffer[-2].postag,
			'b1lb2l'    + self.buffer[-1].lemma + self.buffer[-2].lemma,
			'b1pb2p'    + self.buffer[-1].postag + self.buffer[-2].postag

		)

		index_lcs1_c = self.arcs.left_most_child(self.item_2_index[self.stack[-1]])
		index_rcs1_c = self.arcs.right_most_child(self.item_2_index[self.stack[-1]])
		index_lcs2_c = self.arcs.left_most_child(self.item_2_index[self.stack[-2]])
		index_rcs2_c = self.arcs.right_most_child(self.item_2_index[self.stack[-2]])

		index_lcs1_p = self.arcs.left_most_parent(self.item_2_index[self.stack[-1]])
		index_rcs1_p = self.arcs.right_most_parent(self.item_2_index[self.stack[-1]])
		index_lcs2_p = self.arcs.left_most_parent(self.item_2_index[self.stack[-2]])
		index_rcs2_p = self.arcs.right_most_parent(self.item_2_index[self.stack[-2]])

		# three-word
		three_word_features = (
			's2ls1lb1l'   + self.stack[-2].lemma + self.stack[-1].lemma + self.buffer[-1].lemma,
			's2ps1lb1l'   + self.stack[-2].postag + self.stack[-1].lemma + self.buffer[-1].lemma,
			's2ls1pb1l'   + self.stack[-2].lemma + self.stack[-1].postag + self.buffer[-1].lemma,
			's2ps1pb1l'   + self.stack[-2].postag + self.stack[-1].postag + self.buffer[-1].lemma,
			's2ls1lb1p'   + self.stack[-2].lemma + self.stack[-1].lemma + self.buffer[-1].postag,
			's2ps1lb1p'   + self.stack[-2].postag + self.stack[-1].lemma + self.buffer[-1].postag,
			's2ls1pb1p'   + self.stack[-2].lemma + self.stack[-1].postag + self.buffer[-1].postag,
			's2ps1pb1p'   + self.stack[-2].postag + self.stack[-1].postag + self.buffer[-1].postag,
			

			's2ps1plcs1cp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_lcs1_c].postag,
			's2ps1prcs1cp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_rcs1_c].postag,
			's2ps1plcs2cp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_lcs2_c].postag,
			's2ps1prcs2cp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_rcs2_c].postag,
			's2ps1llcs1cp' + self.stack[-2].postag + self.stack[-1].lemma + self.index_2_item[index_lcs1_c].postag,
			's2ps1lrcs1cp' + self.stack[-2].postag + self.stack[-1].lemma + self.index_2_item[index_rcs1_c].postag,

			's2ps1plcs1pp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_lcs1_p].postag,
			's2ps1prcs1pp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_rcs1_p].postag,
			's2ps1plcs2pp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_lcs2_p].postag,
			's2ps1prcs2pp' + self.stack[-2].postag + self.stack[-1].postag + self.index_2_item[index_rcs2_p].postag,
			's2ps1llcs1pp' + self.stack[-2].postag + self.stack[-1].lemma + self.index_2_item[index_lcs1_p].postag,
			's2ps1lrcs1pp' + self.stack[-2].postag + self.stack[-1].lemma + self.index_2_item[index_rcs1_p].postag,

			'b1ps1plcs1cp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_lcs1_c].postag,
			'b1ps1prcs1cp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_rcs1_c].postag,
			'b1ps1plcs2cp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_lcs2_c].postag,
			'b1ps1prcs2cp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_rcs2_c].postag,
			'b1ps1llcs1cp' + self.buffer[-1].postag + self.stack[-1].lemma + self.index_2_item[index_lcs1_c].postag,
			'b1ps1lrcs1cp' + self.buffer[-1].postag + self.stack[-1].lemma + self.index_2_item[index_rcs1_c].postag,

			'b1ps1plcs1pp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_lcs1_p].postag,
			'b1ps1prcs1pp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_rcs1_p].postag,
			'b1ps1plcs2pp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_lcs2_p].postag,
			'b1ps1prcs2pp' + self.buffer[-1].postag + self.stack[-1].postag + self.index_2_item[index_rcs2_p].postag,
			'b1ps1llcs1pp' + self.buffer[-1].postag + self.stack[-1].lemma + self.index_2_item[index_lcs1_p].postag,
			'b1ps1lrcs1pp' + self.buffer[-1].postag + self.stack[-1].lemma + self.index_2_item[index_rcs1_p].postag
		)

		# history
		history_features = (
			'h1' + self.history[-1],
			'h2' + self.history[-2],
			'h3' + self.history[-3],
			'h1h2' + self.history[-1] + self.history[-2],
			#'h2h3' + self.history[-2] + self.history[-3],
			'h1h2h3' + self.history[-1] + self.history[-2] + self.history[-3]
		)

		return single_word_features + word_pair_features + three_word_features + history_features
		# return history_features + single_word_features

	def shift(self):
		if not self.buffer_empty():
			self.update_history_ok = True
			self.stack.append(self.buffer.pop())

	def reduce(self):
		if not self.stack_empty():
			self.update_history_ok = True
			self.stack.pop()

	def left_arc(self, relation):
		left_item  = self.stack[-1]
		right_item = self.buffer[-1]

		updated = self.arcs.left_arc(self.item_2_index[left_item], self.item_2_index[right_item], relation)

		if updated:
			self.update_history_ok = True

	def right_arc(self, relation):
		left_item  = self.stack[-1]
		right_item = self.buffer[-1]

		updated = self.arcs.right_arc(self.item_2_index[left_item], self.item_2_index[right_item], relation)

		if updated:
			self.update_history_ok = True

	def root(self):
		left_item  = self.stack[-1]
		right_item = self.buffer[-1]

		self.arcs.right_arc(self.item_2_index[left_item], self.item_2_index[right_item], 'root')
		self.reduce()




def main():
	pass

if __name__ == "__main__":
	main()