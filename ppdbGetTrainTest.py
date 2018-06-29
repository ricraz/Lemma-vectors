import random
import os

with open('ppdbLargeFiltered.txt', 'r') as ppdbFiltered:
    first = True
    phrasePairs = []
    secondPhrases = []
    for line in ppdbFiltered:
        if first:
            phrasePairs.append(line)
            phrasePairs.append(line)
        else:
            secondPhrases.append(line)
        first = not first
    randomPhrases = secondPhrases.copy()
    #random.shuffle(randomPhrases)
    for i in range(len(secondPhrases)):
        one = phrasePairs[2*i][:-1].split()
        two = secondPhrases[i][:-1].split()
        if len(one) > len(two):
            one, two = two, one
        phrasePairs[2*i] = (" ".join(one) + os.linesep, " ".join(two) + os.linesep, '1')
        one = phrasePairs[2*i+1][:-1].split()
        two = randomPhrases[i][:-1].split()
        if len(one) > len(two):
            #random.shuffle(one)
            one, two = two, one
        #else:
            #random.shuffle(two)
        phrasePairs[2*i+1] = (" ".join(one) + os.linesep, " ".join(two) + os.linesep, '0')
    random.shuffle(phrasePairs)
    cutoff = len(phrasePairs)*9//10

with open('ppdbShuffledTrain.txt', 'w') as trainFile:
    for item in phrasePairs[:cutoff]:
        trainFile.write(item[0])
        trainFile.write(item[1])
        trainFile.write(item[2] + os.linesep)
with open('ppdbShuffledTest.txt', 'w') as testFile:
    for item in phrasePairs[cutoff:]:
        testFile.write(item[0])
        testFile.write(item[1])
        testFile.write(item[2] + os.linesep)
