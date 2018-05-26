import numpy as np

vectorLength = 100

tags = {'PDT': 'PT', '.':'.','VB':'VB', ':':':', '#':'#', 'VBN':'VN', 'RBR': 'RR', 'PRP$':'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC':'CC', 'TO':'TO', 'LS': 'LS', 'SYM':'SM', 'RBS': 'RS', 'JJ':'JJ', 'EX':'EX', 'WP':'WP', 'POS':'PS', 'WDT':'WT', 'VBP':'VP', 'WRB':'WB', 'PRP':'PP', 'JJR':'JR', 'VBD':'VD', 'NNPS':'NQ', 'RB':'RB','-LRB':'L$', 'RP':'RP', 'JJS':'JS', 'CD':'CD', '-RRB-':'R$', 'NNP': 'NP', '$':'$', 'WP$':'WP', 'FW':'FW', 'VBG':'VG',"''":"''", ',':',', 'NN':'NN','UH':'UH', 'NNS':'NS', 'MD':'MD', '``':'``','IN':'IN'}
with open('../fastText/unlemmaTagSkipWordVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

count = 0

missingWords = set()
with open('missingPPDBUnlemmaTagWords.txt', 'w') as writeFile:
        with open('ppdbLargeFilteredUnlemmaTagTrain.txt', 'r') as wordFile:
                size = int(wordFile.readline())
                for i in range(size):
                        splits = wordFile.readline()[:-1].split()
                        for tempWord in splits:
                                if tempWord not in vectors and tempWord not in missingWords:
                                        missingWords.add(tempWord)
                                        writeFile.write(tempWord + '\n')
                                        if count % 1000 == 0:
                                                print(tempWord)
                                        count += 1
            		
                        splits = wordFile.readline()[:-1].split()
                        for tempWord in splits:
                                if tempWord not in vectors and tempWord not in missingWords:
                                        missingWords.add(tempWord)
                                        writeFile.write(tempWord + '\n')
                                        if count % 1000 == 0:
                                                print(tempWord)
                                        count += 1
                        wordFile.readline()

        with open('ppdbLargeFilteredUnlemmaTagTest.txt', 'r') as wordFile:
                size = int(wordFile.readline())
                for i in range(size):
                        splits = wordFile.readline()[:-1].split()
                        for tempWord in splits:
                                if tempWord not in vectors and tempWord not in missingWords:
                                        missingWords.add(tempWord)
                                        writeFile.write(tempWord + '\n')
                                        if count % 1000 == 0:
                                                print(tempWord)
                                        count += 1

                        splits = wordFile.readline()[:-1].split()
                        for tempWord in splits:
                                if tempWord not in vectors and tempWord not in missingWords:
                                        missingWords.add(tempWord)
                                        writeFile.write(tempWord + '\n')
                                        if count % 1000 == 0:
                                                print(tempWord)
                                        count += 1
                        wordFile.readline()
