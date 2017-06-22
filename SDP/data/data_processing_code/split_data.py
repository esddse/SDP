# coding:utf-8

train = []
dev = []
with open('./dm.sdp','r', encoding='utf8') as f:
	items = f.read().split('\n\n')
	for item in items:
		if item.startswith('#200'):
			train.append(item)
		else:
			dev.append(item)

with open('train','w', encoding='utf8') as f:
	for item in train:
		f.write(item+'\n\n')

with open('dev', 'w', encoding='utf8')  as f:
	for item in dev:
		f.write(item+'\n\n')