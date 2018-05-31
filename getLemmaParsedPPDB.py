from pynlp import StanfordCoreNLP
import random

annotators = 'pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

source = open('ppdbShuffledTest.txt','r')
with open('ppdbShuffledUnlemmaUntaggedTest.txt', 'w') as writeOriginal:
	with open('ppdbShuffledUnlemmaTagTest.txt', 'w') as writeFile:
		with open('ppdbShuffledLemmaUntaggedTest.txt', 'w') as writeFile2:
			with open('ppdbShuffledLemmaTagTest.txt', 'w') as writeFile3:
				with open('ppdbShuffledTagsTest.txt', 'w') as writeFile4:
					length = source.readline()
					writeOriginal.write(length)
					writeFile.write(length)
					writeFile2.write(length)
					writeFile3.write(length)
					writeFile4.write(length)
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

						write0 = []
						write1 = []
						write2 = []
						write3 = []
						write4 = []

						for token in doc1[0]:
							write0.append(str(token).lower())
							write1.append(str(token).lower() + "_" + tags[token.pos])
							write2.append(token.lemma.lower())
							write3.append(token.lemma.lower() + "_" + tags[token.pos])
							write4.append(tags[token.pos])
							
						writeOriginal.write(" ".join(write0)+'\n')
						writeFile.write(" ".join(write1)+'\n')
						writeFile2.write(" ".join(write2)+'\n')
						writeFile3.write(" ".join(write3)+'\n')
						writeFile4.write(" ".join(write4)+'\n')

						write0 = []
						write1 = []
						write2 = []
						write3 = []
						write4 = []

						permute = [i for i in range(len(doc2))]
						if int(third) == 0:
							random.shuffle(permute)
						for item in permute:
							try:
								token = doc2[0][item]
							except IndexError:
								token = doc2[1][item-len(doc2[0])]
							write0.append(str(token).lower())
							write1.append(str(token).lower() + "_" + tags[token.pos])
							write2.append(token.lemma.lower())
							write3.append(token.lemma.lower() + "_" + tags[token.pos])
							write4.append(tags[token.pos])

						writeOriginal.write(" ".join(write0)+'\n')
						writeFile.write(" ".join(write1)+'\n')
						writeFile2.write(" ".join(write2)+'\n')
						writeFile3.write(" ".join(write3)+'\n')
						writeFile4.write(" ".join(write4)+'\n')

						writeOriginal.write(third)
						writeFile.write(third)
						writeFile2.write(third)
						writeFile3.write(third)
						writeFile4.write(third)
						if i % 100 == 0:
							print(i,first, second, write0, write1)
source.close()
