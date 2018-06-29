from pynlp import StanfordCoreNLP
import random

annotators = 'pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})


tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}



with open('ppdb_all2.txt', 'r') as ppdb:
    with open('lemmaTagPPDBTrain.txt', 'w') as writeF1:
        with open('lemmaUntaggedPPDBTrain.txt', 'w') as writeF2:
            with open('unlemmaTagPPDBTrain.txt', 'w') as writeF3:
                with open('unlemmaUntaggedPPDBTrain.txt','w') as writeF4:    
                    writeF1.write('5400'+'\n')
                    writeF2.write('5400'+'\n')
                    writeF3.write('5400'+'\n')
                    writeF4.write('5400'+'\n')
                    
                    first = []
                    second = []
                    third = []
                    for i in range(3000):
                        first.append(ppdb.readline()[:-1])
                        second.append(ppdb.readline()[:-1])
                        third.append(ppdb.readline()[:-1])
                        first.append(second[-1])
                        second.append(first[-1])
                        third.append(third[-1])
                    myList = [i for i in range(6000)]
                    random.shuffle(myList)

                    for i in myList[:5400]:
                        print(i)
                        doc1 = nlp(first[i])
                        doc2 = nlp(second[i])
                        score = third[i]
                        
                        write1 = []
                        write2 = []
                        write3 = []
                        write4 = []

                        for token in doc1[0]:
                            write1.append(token.lemma.lower() + "_" + tags[token.pos])
                            write2.append(token.lemma.lower())
                            write3.append(str(token).lower() + "_" + tags[token.pos])
                            write4.append(str(token).lower())

                        writeF1.write(" ".join(write1)+'\n')
                        writeF2.write(" ".join(write2)+'\n')
                        writeF3.write(" ".join(write3)+'\n')
                        writeF4.write(" ".join(write4)+'\n')

                        write1 = []
                        write2 = []
                        write3 = []
                        write4 = []

                        for token in doc2[0]:
                            write1.append(token.lemma.lower() + "_" + tags[token.pos])
                            write2.append(token.lemma.lower())
                            write3.append(str(token).lower() + "_" + tags[token.pos])
                            write4.append(str(token).lower())

                        writeF1.write(" ".join(write1)+'\n')
                        writeF2.write(" ".join(write2)+'\n')
                        writeF3.write(" ".join(write3)+'\n')
                        writeF4.write(" ".join(write4)+'\n')

                        writeF1.write(third[i]+'\n')
                        writeF2.write(third[i]+'\n')
                        writeF3.write(third[i]+'\n')
                        writeF4.write(third[i]+'\n')

                    with open('lemmaTagPPDBTest.txt', 'w') as writeFT1:
                        with open('lemmaUntaggedPPDBTest.txt', 'w') as writeFT2:
                            with open('unlemmaTagPPDBTest.txt', 'w') as writeFT3:
                                with open('unlemmaUntaggedPPDBTest.txt','w') as writeFT4:    
                                    writeFT1.write('600'+'\n')
                                    writeFT2.write('600'+'\n')
                                    writeFT3.write('600'+'\n')
                                    writeFT4.write('600'+'\n')
                                    for i in myList[5400:]:
                                        print(i)
                                        doc1 = nlp(first[i])
                                        doc2 = nlp(second[i])
                                        score = third[i]
                        
                                        write1 = []
                                        write2 = []
                                        write3 = []
                                        write4 = []

                                        for token in doc1[0]:
                                            write1.append(token.lemma.lower() + "_" + tags[token.pos])
                                            write2.append(token.lemma.lower())
                                            write3.append(str(token).lower() + "_" + tags[token.pos])
                                            write4.append(str(token).lower())

                                        writeFT1.write(" ".join(write1)+'\n')
                                        writeFT2.write(" ".join(write2)+'\n')
                                        writeFT3.write(" ".join(write3)+'\n')
                                        writeFT4.write(" ".join(write4)+'\n')

                                        write1 = []
                                        write2 = []
                                        write3 = []
                                        write4 = []

                                        for token in doc2[0]:
                                            write1.append(token.lemma.lower() + "_" + tags[token.pos])
                                            write2.append(token.lemma.lower())
                                            write3.append(str(token).lower() + "_" + tags[token.pos])
                                            write4.append(str(token).lower())

                                        writeFT1.write(" ".join(write1)+'\n')
                                        writeFT2.write(" ".join(write2)+'\n')
                                        writeFT3.write(" ".join(write3)+'\n')
                                        writeFT4.write(" ".join(write4)+'\n')

                                        writeFT1.write(third[i]+'\n')
                                        writeFT2.write(third[i]+'\n')
                                        writeFT3.write(third[i]+'\n')
                                        writeFT4.write(third[i]+'\n')

