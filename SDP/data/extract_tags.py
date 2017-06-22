
tags = []
with open('./arc_eager_operations', 'r', encoding='utf8') as f:
	for line in f:
		line = line.strip().split(' ')
		for tag in line:
			if tag not in tags:
				tags.append(tag)

with open('./tags', 'w', encoding='utf8') as f:
	for tag in tags:
		f.write(tag + '\n')