with open('msr_paraphrase_train.txt', 'r', encoding='utf8') as file:
	firstOut = open('msrUnlemmaUntaggedTrain.txt', 'w')
	secondOut = open('msrLemmaUntaggedTrain.txt', 'w')
	thirdOut = open('msrLemmaTaggedTrain.txt', 'w')
	file.readline()
	for line in file:
		splitLine = line[:-1].split('\t')
		sentence1 = 
