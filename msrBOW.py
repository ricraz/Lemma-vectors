import numpy as np
from sklearn import svm

vectorLength = 100
separator = [0]*vectorLength

def cosineDistance(vec1,vec2):
    if np.linalg.norm(vec1)*np.linalg.norm(vec2) == 0:
        print(vec1, vec2)
        return 0
    else:
        return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

with open('../fastText/unlemmaUntaggedPuncSkipVectors.vec', 'r') as vectorFile:
    vectors = dict()
    vectorFile.readline()
    for line in vectorFile:
        splitLine = line.split()
        start = len(splitLine) - vectorLength
        vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)
"""
with open('../fastText/missingUnlemmaUntaggedSkipVectors.txt', 'r') as missingFile:
    for line in missingFile:
        splitLine = line.split()
        start = len(splitLine) - vectorLength
        vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)
"""
with open('ppdbLargeFilteredTrain.txt', 'r') as msrFile:
    length = int(msrFile.readline())
    x_train = []
    y_train = []
    for i in range(length):
        first = msrFile.readline()[:-1].split()
        second = msrFile.readline()[:-1].split()
        third = int(msrFile.readline())
        firstVector = np.zeros(vectorLength)
        secondVector = np.zeros(vectorLength)
        for item in first:
            if item in vectors:
                firstVector += vectors[item]
        for item in second:
            if item in vectors:
                secondVector += vectors[item]
        
        x_train.append([cosineDistance(firstVector, secondVector)])
        y_train.append(third)

model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

with open('ppdbLargeFilteredLemmaTest.txt', 'r') as msrFile:
    length = int(msrFile.readline())
    x_test = []
    y_test = []
    for i in range(length):
        first = msrFile.readline()[:-1].split()
        second = msrFile.readline()[:-1].split()
        third = int(msrFile.readline())
        firstVector = np.zeros(vectorLength)
        secondVector = np.zeros(vectorLength)
        for item in first:
            if item in vectors:
                firstVector += vectors[item]
        for item in second:
            if item in vectors:
                secondVector += vectors[item]
        
        x_test.append([cosineDistance(firstVector, secondVector)])
        y_test.append(third)

y_pred = model.predict(x_test)
outcome = sum([y_pred[i] == y_test[i] for i in range(length)])/length
print(outcome)
