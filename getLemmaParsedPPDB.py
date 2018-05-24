from pynlp import StanfordCoreNLP

annotators = 'pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

with open('ppdbLargeFilteredTrain.txt', 'r') as source:
	with open('ppdbLargeFilteredUnlemmaTagTrain.txt', 'w') as writeFile:
		with open('ppdbLargeFilteredLemmaTrain2.txt', 'w') as writeFile2:
			length = source.readline()
			writeFile.write(length)
			writeFile2.write(length)
			for i in range(int(length)):
				first = source.readline()[:-1]
				second = source.readline()[:-1]
				third = source.readline()
				first = first.replace("-rrb-",")")
				first = first.replace("-lrb-","(")
				second = second.replace("-rrb-",")")
				second = second.replace("-lrb-","(")

				doc1 = nlp(first)
				doc2 = nlp(second)

				write = []
				write2 = []

				for token in doc1[0]:
					write.append(str(token).lower() + "_" + tags[token.pos])
				#	write2.append(token.lemma.lower())
				writeFile.write(" ".join(write)+'\n')
				#writeFile2.write(" ".join(write2)+'\n')

				write = []
				write2 = []
				for token in doc2[0]:
					write.append(str(token).lower() + "_" + tags[token.pos])
				#	write2.append(token.lemma.lower())
				writeFile.write(" ".join(write)+'\n')
				#writeFile2.write(" ".join(write2)+'\n')
				
				writeFile.write(third)
				#writeFile2.write(third)
				if i % 100 == 0:
					print(i,first, second, write)
