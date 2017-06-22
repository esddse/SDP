# coding:utf-8

from model import *
from util import *

import pickle
import sys
import os
import numpy as np

TRAIN_EPOCH = 1


def random_shuffle(datas, operations):
	data_size = len(datas)
	shuffled_indices = np.random.permutation(np.arange(data_size))

	shuffled_datas = [0] * data_size
	shuffled_operations = [0] * data_size

	for position, index in zip(range(data_size), shuffled_indices):
		shuffled_datas[position] = datas[index]
		shuffled_operations[position] = operations[index]
	return shuffled_datas, shuffled_operations 


def train():

	# read data
	train_datas = list(map(data_transform, read_train_data('./data/dm.sdp')))
	train_operations = read_train_operations('./data/dm.sdp.operations')
	data_size = len(train_datas)

	operations = read_operations()
	perceptron = Perceptron(operations)


	shuffled_datas, shuffled_operations = random_shuffle(train_datas, train_operations)

	slice_index = int(data_size*0.05)
	train_xs, train_ys = shuffled_datas[slice_index:], shuffled_operations[slice_index:]
	dev_xs, dev_ys = shuffled_datas[:slice_index], shuffled_operations[:slice_index]


	for epoch in range(TRAIN_EPOCH):
		os.system('rm ./dev_golden ./dev_predict')
		
		train_xs, train_ys = random_shuffle(train_xs, train_ys)
		
		# training
		# ================================================================
		sentence_cnt = 0
		total_acc = 0
		total_arc = 0
		for data, op_seq in zip(train_xs, train_ys):
			number = data[0]
			data = data[1]
			# init configuration
			sentence = []
			for index in range(len(data)):
				sentence.append(ConfigurationItem(data[index]['form'], data[index]['lemma'], data[index]['postag']))

			configuration = Configuration(sentence)
		
			
			predictions = []

			acc_cnt = 0
			op_index = 0
			while (not configuration.stack_empty() or not configuration.buffer_empty()) and configuration.history_size() < len(op_seq):
				if op_seq[op_index] != 'REDUCE' and op_seq[op_index] != 'SHIFT':
					total_arc += 1

				# predict
				prediction = perceptron.predict(configuration)
				# update perceptron
				if prediction != op_seq[op_index]:
					perceptron.update(configuration, op_seq[op_index], 1)
					perceptron.update(configuration, prediction, -1)
				else:
					if prediction != 'REDUCE' and prediction != 'SHIFT':
						acc_cnt += 1


				if op_seq[op_index] == 'SHIFT':
					configuration.shift()
				elif op_seq[op_index] == 'REDUCE':
					configuration.reduce()
				elif op_seq[op_index] == 'ROOT':
					configuration.root()
				else:
					op = op_seq[op_index].split('__')
					relation = op[0]
					direction = op[1]

					if direction == 'LEFT':
						configuration.left_arc(relation)
					else:
						configuration.right_arc(relation)

				# update configuration
				if configuration.update_history_ok:
					predictions.append(prediction)
					configuration.history.append(op_seq[op_index])
					configuration.update_history_ok = False
					op_index += 1
				
				#configuration.print_state()
				#a = input()

			total_acc += acc_cnt 

			sentence_cnt += 1
			print('sentence:', sentence_cnt, ' accuracy = ',total_acc/total_arc, '  weights_size = ', len(perceptron.weights))


		# perceptron average
		perceptron.average()
		print('saving model to ./model/perceptron_0')
		save_model(perceptron, './model/perceptron_0')

		# evaluation
		# ================================================================
		print('start evaluation...')
		for data, op_seq in zip(dev_xs, dev_ys):
			number = data[0]
			data = data[1]
			# init configuration
			sentence = []
			for index in range(len(data)):
				sentence.append(ConfigurationItem(data[index]['form'], data[index]['lemma'], data[index]['postag']))

			configuration = Configuration(sentence)

			while (not configuration.stack_empty() or not configuration.buffer_empty()):
				# predict
				prediction = perceptron.predict(configuration)

				if prediction == 'SHIFT':
					configuration.shift()
				elif prediction == 'REDUCE':
					configuration.reduce()
				elif prediction == 'ROOT':
					configuration.root()
				else:
					op = prediction.split('__')
					relation = op[0]
					direction = op[1]

					if direction == 'LEFT':
						configuration.left_arc(relation)
					else:
						configuration.right_arc(relation)

				# update configuration
				if configuration.update_history_ok:
					configuration.history.append(prediction)
					configuration.update_history_ok = False
		
	
		
			# write to file
			with open('./dev_golden', 'a', encoding='utf8') as f:
				f.write(number+'\n')

				for i in range(len(data)):
					f.write(str(i+1))
					f.write('\t'+data[i]['form'])
					f.write('\t'+data[i]['lemma'])
					f.write('\t'+data[i]['postag'])
					f.write('\t'+data[i]['top'])
					f.write('\t'+data[i]['pred'])
					for arg in data[i]['arg']:
						f.write('\t'+arg)
					f.write('\n')
				f.write('\n')

			configuration.arcs.format_output(data)
			with open('./dev_predict', 'a', encoding='utf8') as f:
				f.write(number+'\n')

				for i in range(len(data)):
					f.write(str(i+1))
					f.write('\t'+data[i]['form'])
					f.write('\t'+data[i]['lemma'])
					f.write('\t'+data[i]['postag'])
					f.write('\t'+data[i]['ptop'])
					f.write('\t'+data[i]['ppred'])
					for arg in data[i]['parg']:
						f.write('\t'+arg)
					f.write('\n')
				f.write('\n')


		result = os.popen('./evalue.sh').read()
		print(result)



			



def main():
	train()



if __name__ == "__main__":
	main()




