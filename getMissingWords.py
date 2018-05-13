import numpy as np

vectorLength = 100
from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer = WordNetLemmatizer()

tags = {'PDT': 'PT', '.':'.','VB':'VB', ':':':', '#':'#', 'VBN':'VN', 'RBR': 'RR', 'PRP$':'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC':'CC', 'TO':'TO', 'LS': 'LS', 'SYM':'SM', 'RBS': 'RS', 'JJ':'JJ', 'EX':'EX', 'WP':'WP', 'POS':'PS', 'WDT':'WT', 'VBP':'VP', 'WRB':'WB', 'PRP':'PP', 'JJR':'JR', 'VBD':'VD', 'NNPS':'NQ', 'RB':'RB','-LRB':'L$', 'RP':'RP', 'JJS':'JS', 'CD':'CD', '-RRB-':'R$', 'NNP': 'NP', '$':'$', 'WP$':'WP', 'FW':'FW', 'VBG':'VG',"''":"''", ',':',', 'NN':'NN','UH':'UH', 'NNS':'NS', 'MD':'MD', '``':'``','IN':'IN'}
with open('../fastText/lemmaTagSkipWordVectors.vec', 'r') as vectorFile:
	vectors = set()
	i = 0
	vectorFile.readline()
	for phrase in vectorFile:
		listing = phrase.split()
		length = len(listing)
		try:
			vectors.add(" ".join(listing[0:length-vectorLength]))
			#vectors[" ".join(listing[0:length-vectorLength])] = np.array(listing[length - vectorLength:], dtype=float)
		except ValueError:
			print(phrase)
			print(listing)
			print(len(listing))
			vectors.add(" ".join(listing[0:length-vectorLength]))
		if i % 10000 == 0:
			print(listing[0])
		i+=1
count = 0
missingWords = set()
with open('../ppdbLargeFiltered.txt', 'r') as wordFile:
	with open('missingLemmaTaggedWords.txt', 'w') as writeFile:
		for phrase in wordFile:
			text = phrase.split()
			text = nltk.pos_tag(text)
			for word in text:
				try:
					tempWord = lemmatizer.lemmatize(word[0]).lower() + "_" + tags[word[1]]
				except KeyError:
					print(text)
					print(word)
				if tempWord not in vectors and tempWord not in missingWords:
					missingWords.add(tempWord)
					writeFile.write(tempWord + '\n')
					if count % 1000 == 0:
						print(tempWord)
					count += 1
