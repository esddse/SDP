
from model import *
from util import *


def train():

	# -----------------init model -------------------
	# -----------------------------------------------
	test_config = config()
	sdp_parser = stack_lstm_parser(test_config)

	# ----------------- load data -------------------
	# -----------------------------------------------
	train_datas = list(map(data_transform, read_train_data('./data/dm.sdp')))
	train_operations = read_train_operations('./data/dm.sdp.operations')

	
	for word_list, operation_list in zip(train_datas, train_operations):
		# loss, accuracy, arcs, root, predictions = sdp_parser.full_step(word_list, operation_list, train=True)
		loss, accuracy, arcs, root, predictions = sdp_parser.full_step(train_datas[0], train_operations[0], train=True)

		print('loss: ', loss, '  acc: ', accuracy, '    len_pred: ', len(predictions), '  len_ys: ', len(train_operations[0]))
		print(predictions)


def main():
	train()

if __name__ == '__main__':
	main()