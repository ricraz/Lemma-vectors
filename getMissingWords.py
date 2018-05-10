import numpy as np

vectorLength = 100

with open('fastText/lemmaVectors.vec', 'r') as vectorFile:
	vectors = dict()
	i = 0
	vectorFile.readline()
	for phrase in vectorFile:
		listing = phrase.split()
		length = len(listing)
		try:
			vectors[" ".join(listing[0:length-vectorLength])] = np.array(listing[length - vectorLength:], dtype=float)
		except ValueError:
			print(phrase)
			print(listing)
			print(len(listing))
			vectors[listing[0]] = np.array(listing[1:], dtype=float)
		if i % 10000 == 0:
			print(listing[0], vectors[listing[0]])
		i+=1
count = 0
missingWords = set()
with open('ppdbLargeFiltered.txt', 'r') as wordFile:
	with open('missingWords.txt', 'w') as writeFile:
		for phrase in wordFile:
			for word in phrase.split():
				if word not in vectors and word not in missingWords:
					missingWords.add(word)
					writeFile.write(word + '\n')
					if count % 10000 == 0:
						print(word)
					count += 1			
