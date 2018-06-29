'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
vectorLength = 100
global maxLength
maxLength = 6
batch_size = 32

with open('fastText/lemmaVectors.vec', 'r') as file:
    vectors = dict()
    file.readline()
    for line in file:
        splitLine = line.split()
        start = len(splitLine) - vectorLength
        try:
            vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:]).astype(float)
        except ValueError:
            print(splitLine, len(splitLine))
print(len(vectors))
i = 0
for key, value in vectors.items():
    print(key, value, i)
    i += 1
    if i > 5:
        break
print("just did it")

def getVector(line, vectors):
    output = []
    split = line.split()
    for word in split:
        tempWord = lemmatizer.lemmatize(word.lower())
        if tempWord in vectors:
            output.append(vectors[tempWord])
        else:
            output.append([0]*vectorLength)
    for i in range(maxLength - len(output)):
        output.append([0]*vectorLength)
    return output

print(getVector("hello world , my name is Richard", vectors))
print('Loading data...')
outfile = open("lemma_ppdb_large_filtered.txt", 'w')
outfile.write("MaxLength = " + str(maxLength) + " Vector length = " + str(vectorLength) +  " test = 10%")


with open("ppdbLargeFiltered.txt", 'r') as ppdb:
    first = True
    phrasePairs = []
    secondPhrases= []
    
    for line in ppdb:
        if first:
            current = getVector(line, vectors)
            
            phrasePairs.append(current)
            phrasePairs.append(current)
        else:
            secondPhrases.append(getVector(line, vectors))
        first = not first

    randomPhrases = secondPhrases.copy()
    random.shuffle(randomPhrases)
    for i in range(len(secondPhrases)):
        phrasePairs[2*i] = ((phrasePairs[2*i] + secondPhrases[i]),1)
        phrasePairs[2*i+1] = ((phrasePairs[2*i+1] + randomPhrases[i]),0)

    random.shuffle(phrasePairs)

    x_train, y_train = zip(*phrasePairs)
    #x_train = np.array(x_train)
    #y_train = np.array(y_train)
    for i in range(len(x_train)):
        outfile.write(str(x_train[i]))
        outfile.write(str(y_train[i]))
    cutoff = len(x_train)*9//10
    x_train, x_test = x_train[:cutoff], x_train[cutoff:]
    y_train, y_test = y_train[:cutoff], y_train[cutoff:]

outfile.close()

print(x_train[:2], y_train[:10])
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
"""
x_train = np.array([np.array([[1,2],[2,2],[2,3]]),np.array([[2,6],[2,9]]),np.array([[2,3]])])
x_train = [[[1,1]],[[1,2],[2,2],[2,3]],[[1,3],[1,1]]]
#print(x_train.shape)
#for item in x_train:
#    print(item.shape)
#x_train.reshape(3,2,1)
print(x_train)
#print(x_train.shape)
y_train = [0,1,0]
x_test = x_train.copy()
y_test = y_train.copy()
"""
print('Build model...')
model = Sequential()
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(None,2)))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
