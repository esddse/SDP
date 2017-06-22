import sys
import copy
sys.path.append('../../')

from util import *
 

class operation_generator(object):


	def __init__(self):
		self.stack = ['ROOT']
		self.buffer = []

		# recorder to record operations
		self.recorder = []

	def clear(self):
		self.stack = ['ROOT']
		self.buffer = []

		self.recorder = []

	def stack_empty(self):
		return True if len(self.stack) == 0 else False

	def buffer_empty(self):
		return True if len(self.buffer) == 0 else False

	def print_stack(self):
		print('stack:  ' ,end='')
		for item in self.stack:
			if item == 'ROOT':
				print(item, ' ', end='')
			else:
				print(item['form'], ' ', end='')
		print()

	def print_buffer(self):
		print('buffer: ' ,end='')
		for item in self.buffer:
			print(item['form'], ' ', end='')
		print()

	def print_record(self):
		print('record: ', end='')
		for record in self.recorder:
			print(record, ' ', end='')
		print()

	def print_state(self):
		self.print_stack()
		self.print_buffer()
		self.print_record()
		print()

	def fill_in_buffer(self, sequence):
		for item in sequence:
			self.buffer.append(item)
		self.buffer.reverse()

		# return the length of the buffer 
		return len(self.buffer)
	
	# ------------ 4 main operations in arc-eager -----------------
	# -------------------------------------------------------------

	# a basic assumption is: once an item is about to be pushed into the stack,
	# all the connections between this item and items in the stack have already considered
	# 

	# shift means the stack_top_item has arc connected to item behind buffer_top_item
	def shift(self):
		self.stack.append(self.buffer.pop())
		self.recorder.append('SHIFT')

	# reduce means that all the arcs that connected to the stack_top_item has been considered 
	def reduce(self):
		self.stack.pop()
		self.recorder.append('REDUCE')

	# stact_top_item <------- buffer_top_item
	# just build the arc
	def left_arc(self):
		item_left  = self.stack[-1]
		item_right = self.buffer[-1]

		# record
		self.recorder.append(item_left['arc_in'][item_right['pred_cnt']]+'__LEFT')

		# eliminate corresponding records in item
		item_left['arc_in'][item_right['pred_cnt']]  = '_'
		item_right['arc_out'][item_left['word_cnt']] = '_'

	# stack_top_item ------> buffer_top_item
	def right_arc(self):
		item_left = self.stack[-1]
		item_right = self.buffer[-1]

		if item_left == 'ROOT':
			self.recorder.append('ROOT')
			self.stack.pop()
			return

		# record 
		self.recorder.append(item_right['arc_in'][item_left['pred_cnt']]+'__RIGHT')

		# eliminate corresponding records in item
		item_left['arc_out'][item_right['word_cnt']] = '_'
		item_right['arc_in'][item_left['pred_cnt']]  = '_'


	# ------------------- generation process --------------------
	# -----------------------------------------------------------

	# generate item sequence from data sequence
	# item: a dict
	#   -- form    : the original word
	#   -- top     : whether the root of the sentence 
	#   -- word_cnt: the word number, starts from 0
	#   -- pred_cnt: the predicate number ,starts from 0; if not predicate, -1
	#   -- arc_in  : arc that point to this item, [len] predicate length
	#   -- arc_out : arc that point out from this item [len] sentence length
	def generate_items(self, data):

		sentence_len = len(data)

		# generate arc matrix for every item
		arcs = []
		for word in data:
			arcs.append(word[5])

		# generate items
		item_list = []
		pred_cnt = 0
		word_cnt = 0
		for word in data:
			form  = word[0]
			lemma = word[1]
			pos   = word[2]
			top   = word[3]
			pred  = word[4]
			arg   = word[5]

			item = {}

			# form
			item['form'] = form

			# word_cnt
			item['word_cnt'] = word_cnt
			word_cnt += 1

			# top
			item['top']  = top

			# pred_cnt
			if pred == '+':
				item['pred_cnt'] = pred_cnt
				pred_cnt += 1
			else:
				item['pred_cnt'] = -1

			# arc_in
			item['arc_in'] = arg

			# arc-out
			if pred == '-':
				item['arc_out'] = None
			else:
				arc_out = []
				for arc_in in arcs:
					arc_out.append(arc_in[pred_cnt-1])
				item['arc_out'] = arc_out
			item_list.append(item)

		return item_list

	# check un-processed in_arc
	def check_item_in_arc(self, item):
		cnt = 0
		for tag in item['arc_in']:
			if tag != '_':
				cnt += 1
		return cnt

	# check un-processed out_arc
	def check_item_out_arc(self, item):
		if item['arc_out'] is None:
			return 0
		cnt = 0
		for tag in item['arc_out']:
			if tag != '_':
				cnt += 1
		return cnt

	def generate_operation_sequence(self, data):
		items = self.generate_items(data)
		
		self.fill_in_buffer(items)

		while not self.stack_empty() or not self.buffer_empty():

			if self.stack_empty():
				self.shift()
				continue
			if self.buffer_empty():
				self.reduce()
				continue

			left_item  = self.stack[-1]
			right_item = self.buffer[-1]


			if left_item == 'ROOT':
				# find root
				if right_item['top'] == '+':
					self.right_arc()
				# not root
				else:
					self.shift()

			else:
				# check arcs not processed in the left item
				# reduce situation
				if self.check_item_in_arc(left_item) == 0 and self.check_item_out_arc(left_item) == 0:
					self.reduce()
				
				# left-arc situation
				elif right_item['pred_cnt'] >= 0 and right_item['arc_out'][left_item['word_cnt']] != '_':
					self.left_arc()

				# right-arc situation
				elif left_item['pred_cnt'] >= 0 and left_item['arc_out'][right_item['word_cnt']] != '_':
					self.right_arc()

				else:
					self.shift()

			# self.print_state()

		record = copy.deepcopy(self.recorder)
		self.clear()
		return record



def main():
	op_generator = operation_generator()
	datas = read_train_data('../dm.sdp')

	with open('../arc_eager_operations', 'w') as f:
		# a data is a sentence
		for data in datas:
			data = data_transform(data)
			record = op_generator.generate_operation_sequence(data)

			for operation in record:
				f.write(operation+' ')
			f.write('\n')

		
	

'''
def main():
	datas = read_train_data('../../data/dm.sdp')

	for data in datas:
		form = data[0]
		lemma = data[1]
		pos = data[2]
		top = data[3]
		pred = data[4]
		arg = data[5]

		for i in form:
			print(i,' ', end='')
		print()
		for i in pred:
			print(i, ' ', end = '')
		print()
		
		for i in range(len(form)):
			print("form: ", form[i])
			print("lemma: ", lemma[i])
			print("pos: ", pos[i])
			print("top: ", top[i])
			print("pred: ", pred[i])
			print("arg: ", arg[i])

			a = input()
'''


if __name__ == "__main__":
	main()