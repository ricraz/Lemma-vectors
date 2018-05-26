import csv
import numpy as np
from scipy import spatial
from sklearn import svm
vectorLength = 100

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

with open('../fastText/lemmaTagSkipWordVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

with open('../fastText/missingLemmaTagVectors.txt', 'r') as missingFile:
	for line in missingFile:
		splitLine = line.split()
		start = len(splitLine) - vectorLength
		if " ".join(splitLine[0:start]) not in vectors:
			vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

def getVector(sentence, vectors):
    resultVector = np.zeros(vectorLength)
    splitLine = sentence.split()
    for word in splitLine:
        if word in vectors:
            resultVector = resultVector + vectors[word]
        else:
            print("Problem", word)
            raise(ValueError)
    return resultVector
    
with open('msrLemmaTaggedTrain.txt', 'r', encoding='utf8') as trainFile:
    distances = []
    truths = []
    fileSize = int(trainFile.readline())

    for i in range(fileSize):
        data1 = getVector(trainFile.readline()[:-1], vectors)
        data2 = getVector(trainFile.readline()[:-1], vectors)
        distances.append([spatial.distance.cosine(data1,data2)])
        truths.append(int(trainFile.readline()))

    model = svm.SVC(kernel = 'linear')
    model.fit(distances, truths)
    
    y_pred = model.predict(distances)

    outcome = [truths[i]==y_pred[i] for i in range(len(truths))]

    print("Training accuracy: ", sum(outcome)/fileSize)
    print("Training F1 score: ", sklearn.metrics.f1_score(truths,y_pred))

with open('msrLemmaTaggedTest.txt', 'r', encoding='utf8') as testFile:
    distances = []
    truths = []
    fileSize = int(testFile.readline())
    for i in range(fileSize):
        data1 = getVector(testFile.readline()[:-1], vectors)
        data2 = getVector(testFile.readline()[:-1], vectors)
        distances.append([spatial.distance.cosine(data1,data2)])
        truths.append(int(testFile.readline()))

    y_pred = model.predict(distances)
    outcome = [truths[i]==y_pred[i] for i in range(len(truths))]
    print("Test accuracy: ", sum(outcome)/fileSize)
    print("Training F1 score: ", sklearn.metrics.f1_score(truths,y_pred))
