import csv
import numpy as np
from scipy import spatial
from sklearn import svm
from sklearn import metrics
vectorLength = 100

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}
inverseTags = dict()
for key, item in tags.items():
	inverseTags[item] = key
#missing = open("missingMSRLemmaUntaggedUnhyphenWords.txt",'w')
#missingSet = set()
with open('../vectors/lemmaTagVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

with open('missing/missingLEMMATAGVectors.txt', 'r') as missingFile:
	for line in missingFile:
		splitLine = line.split()
		start = len(splitLine) - vectorLength
		if " ".join(splitLine[0:start]) not in vectors:
			vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)
		else:
			print("Weird!",splitLine)

#with open('../vectors/tagWordVectors.vec','r') as tagFile:
#	tagVectors = dict()
#	tagFile.readline()
#	for line in tagFile:
#		splitLine = line.split()
#		tagVectors[splitLine[0]] = np.array(splitLine[1:], dtype=float)

def getVector(sentence, vectors):
    resultVector = np.zeros(vectorLength)
    tagVector = np.zeros(10)
    splitLine = sentence.split()
    for word in splitLine:
        if word in vectors:
             resultVector = resultVector + vectors[word]
        else:
             print("Woah", word, splitLine)
    return resultVector
#        try:
#            if word[-2] == '_':
#                tempWord = word[:-2]
#                if tempWord in vectors:
#                    resultVector = resultVector + vectors[tempWord]
#                    tagVector = tagVector + tagVectors[inverseTags[word[-1]]]
#            elif word[-3] == '_':
#                tempWord = word[:-3]
#                if tempWord in vectors:
#                    resultVector = resultVector + vectors[tempWord]
#                    tagVector = tagVector + tagVectors[inverseTags[word[-2:]]]
#        except IndexError:
#            pass
#else:
#                print("Cray", word, splitLine)
#        else:
#            print("Woah", word, splitLine)
#    return resultVector
#        else:
#            tempWord = word
#            index = tempWord.find('-')
#            while index != -1:
#                current = tempWord[:index]
#                if current in vectors:
#                    resultVector = resultVector + vectors[current]
               #elif current != "" and current not in missingSet:
                #    print("Problem1", current)
                #    missing.write(current + '\n')
                #    missingSet.add(current)
#                tempWord = tempWord[index+1:]
#                index = tempWord.find('-')
#            if tempWord in vectors:
#                resultVector = resultVector + vectors[tempWord]
            #elif tempWord != "" and tempWord not in missingSet:
            #    print("Problem2", tempWord)
            #    missing.write(tempWord + '\n')
            #    missingSet.add(tempWord)
                #raise(ValueError)"""
#    return resultVector    
#    return np.concatenate((resultVector,tagVector))

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

with open('lemmaTagPPDBTrain.txt', 'r', encoding='utf8') as trainFile:
    distances = []
    truths = []
    fileSize = int(trainFile.readline())

    for i in range(fileSize):
        data1 = getVector(trainFile.readline()[:-1], vectors)
        data2 = getVector(trainFile.readline()[:-1], vectors)
        distances.append([spatial.distance.cosine(data1,data2)])
        truths.append(float(trainFile.readline()))
    
#    cutoff = len(distances)*8//10
#    best = 0
    
    model = linear_model.LinearRegression()
    model.fit(distances, truths)
    #for i in range(-4, 5):
    #    print(i)
    #    model = svm.SVC(kernel = 'linear',C=10**i)
    #    model.fit(distances[:cutoff], truths[:cutoff])
    
    #y_pred = model.predict(distances)

     #   f1 = metrics.f1_score(truths[cutoff:],y_pred[cutoff:])
     #   print(f1)
   #     if f1 > best:
   #         best = f1
   #         cvalue = i
#    model = svm.SVC(kernel='linear', C=10**cvalue)
#    model.fit(distances,truths)

#    outcome = [truths[i]==y_pred[i] for i in range(len(truths))]
    print("training error: ", mean_squared_error(model.predict(distances), truths))
    print("Training Spearman's: ", spearmanr(model.predict(distances),truths))
#    print("Training F1 score: ", metrics.f1_score(truths,y_pred))
#    print(cvalue)

with open('lemmaTagPPDBTest.txt', 'r', encoding='utf8') as testFile:
    distances = []
    truths = []
    fileSize = int(testFile.readline())
    for i in range(fileSize):
        data1 = getVector(testFile.readline()[:-1], vectors)
        data2 = getVector(testFile.readline()[:-1], vectors)
        distances.append([spatial.distance.cosine(data1,data2)])
        truths.append(float(testFile.readline()))

    y_pred = model.predict(distances)
    print("Test error: ", mean_squared_error(y_pred,truths))
    print("Test spearman's: ", spearmanr(y_pred, truths))
	#missing.close()
