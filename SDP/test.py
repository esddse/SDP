# coding:utf-8

from model import *
from util import *

import pickle
import sys
import os
import numpy as np


def test():

	# read data
	test_xs = list(map(data_transform, read_test_data('./data/test')))
	data_size = len(test_xs)

	perceptron = load_model('./model/perceptron_0')

	os.system('rm ./test_predict')

	# testing
	# ================================================================
	for data in test_xs:
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
		

		configuration.arcs.format_output(data)
		with open('./test_predict', 'a', encoding='utf8') as f:
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


def main():
	test()

if __name__ == "__main__":
	main()
