# coding:utf-8

import pickle
import numpy as np

train_path = './data/dm.sdp'
test_path  = './data/test' 

def read_test_data(path):
	datas = []
	with open(path, 'r', encoding='utf8') as f:
		items = f.read().split('\n\n')
		for item in items:
			if item == '':
				continue
			form, lemma, pos = [], [], []
			item = item.strip().split('\n')
			number = item[0]
			lines = item[1:]
			for line in lines:
				line = line.split('\t')
				form.append(line[1])
				lemma.append(line[2])
				pos.append(line[3])
			datas.append([number, form, lemma, pos])
	return datas

def read_train_data(path):
	datas = []
	with open(path, 'r', encoding='utf8') as f:
		items = f.read().split('\n\n')
		for item in items:
			if item == '':
				continue
			form, lemma, pos, top, pred, arg = [], [], [], [], [], []
			item = item.strip().split('\n')
			number = item[0]
			lines = item[1:]
			for line in lines:
				line = line.split('\t')
				form.append(line[1])
				lemma.append(line[2])
				pos.append(line[3])
				top.append(line[4])
				pred.append(line[5])
				arg.append(line[6:])
			datas.append([number, form, lemma, pos, top, pred, arg])
	return datas

def read_train_operations(path):
	operations = []
	with open(path, 'r', encoding='utf8') as f:
		for line in f:
			line = line.strip().split(' ')
			operations.append(line)
	return operations

# transform data to word_based
def data_transform(data):
	number = data[0]
	forms = data[1]
	lemmas = data[2]
	poss = data[3]
	tops = data[4] if len(data) > 4 else None
	preds = data[5] if len(data) > 4 else None
	args = data[6] if len(data) > 4 else None

	word_based_data = []

	for i in range(len(forms)):
		if tops is None:
			item = {
			'form': forms[i],
			'lemma': lemmas[i],
			'postag': poss[i]
		}
		else:
			item = {
				'form': forms[i],
				'lemma': lemmas[i],
				'postag': poss[i],
				'top': tops[i],
				'pred': preds[i],
				'arg': args[i]
			}
		word_based_data.append(item)

	return (number, word_based_data)

def read_word_vector(path):
	word_vector = {}
	with open(path,'r',encoding='utf8') as f:
		# read meta data, size and dim
		meta = list(map(int, f.readline().strip().split(' ')))
		word_vector_size = meta[0]
		word_vector_dim = meta[1]
		
		cnt = 0
		threshold = 0.
		# read word vector
		for line in f:
			line = line.strip().split(' ')
			word_vector[line[0]] = line[1:]

			cnt += 1
			if cnt / word_vector_size > threshold:
				print('loading....... ', cnt / word_vector_size)
				threshold += 0.1

	return word_vector, word_vector_size, word_vector_dim

def read_words():
	with open('./data/config/words', 'r', encoding='utf8') as f:
		return f.read().strip().split('\n')

def read_lemmas():
	with open('./data/config/lemmas', 'r', encoding='utf8') as f:
		return f.read().strip().split('\n')

def read_postags():
	with open('./data/config/postags', 'r', encoding='utf8') as f:
		return f.read().strip().split('\n')

def read_operations():
	with open('./data/config/operations', 'r', encoding='utf8') as f:
		return f.read().strip().split('\n')

def read_relations():
	with open('./data/config/relations', 'r', encoding='utf8') as f:
		return f.read().strip().split('\n')


def gen_data():
	lemmas = []
	poss = []
	tags = []
	datas = read_train_data(train_path)
	for data in datas:
		for lemma in data[2]:
			if lemma not in lemmas:
				lemmas.append(lemma)
		for pos in data[3]:
			if pos not in poss:
				poss.append(pos)
		for args in data[6]:
			for arg in args:
				if arg not in tags:
					tags.append(arg)
	with open('lemmas', 'w', encoding='utf8') as f:
		for lemma in lemmas:
			f.write(lemma+'\n')
	with open('postags', 'w', encoding='utf8') as f:
		for pos in poss:
			f.write(pos+'\n')
	with open('relations', 'w', encoding='utf8') as f:
		for tag in tags:
			f.write(tag+'\n')

def gen_words():
	words = []

	datas = read_train_data(train_path)
	for data in datas:
		for word in data[1]:
			if word not in words:
				words.append(word)

	datas = read_test_data(test_path)
	for data in datas:
		for word in data[1]:
			if word not in words:
				words.append(word)

	with open('./data/config/words', 'w', encoding='utf8') as f:
		for word in words:
			f.write(word+'\n')

def gen_operations():
	operations = []
	with open('./data/dm.sdp.operations', 'r', encoding='utf8') as f:
		for line in f:
			line = line.strip().split(' ')
			for operation in line:
				if operation not in operations:
					operations.append(operation)

	with open('./data/config/operations', 'w', encoding='utf8') as f:
		for operation in operations:
			f.write(operation+'\n')

def gen_relations():
	relations = ['ROOT']
	operations = read_operations()
	for operation in operations:
		if 'LEFT' in operation or 'RIGHT' in operation:
			relations.append(operation.split('__')[0])

	with open('./data/config/relations', 'w', encoding='utf8') as f:
		for relation in relations:
			f.write(relation+'\n')


def get_useful_word_vector():
	words = read_words()

	print('get ',len(words),' words')

	dic = {}

	with open('./model/embeddings/wiki.en.vec','r',encoding='utf8') as f:
		print(f.readline())
		for line in f:
			line = line.strip().split(' ')
			if line[0] in words:
				dic[line[0]] = line[1:]

	oov = []
	for word in words:
		if word not in dic:
			oov.append(word)

	with open('./model/embeddings/oov', 'w', encoding='utf8') as f:
		for word in oov:
			f.write(word+'\n')

	with open('./model/embeddings/useful_embedding', 'w', encoding='utf8') as f:
		for word, vector in dic.items():
			f.write(word)
			for num in vector:
				f.write(' '+num)
			f.write('\n')



def main():
	gen_relations()



	

if __name__ == '__main__':
	main()