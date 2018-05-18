from pynlp import StanfordCoreNLP

annotators = 'tokenize, pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

with open('msr_paraphrase_train.txt', 'r', encoding='utf-8') as file:
        firstOut = open('msrUnlemmaUntaggedTrain.txt', 'w')
        secondOut = open('msrLemmaTaggedTrain.txt', 'w')
        thirdOut = open('msrLemmaUntaggedTrain.txt', 'w')
        file.readline()
        firstOut.write("4076\n")
        secondOut.write("4076\n")
        thirdOut.write("4076\n")
        i = 0
        for line in file:
                splitLine = line[:-1].split('\t')
                doc1 = nlp(splitLine[3].encode("utf-8"))
                doc2 = nlp(splitLine[4].encode("utf-8"))

                write = []
                write2 = []
                write3 = []
                for token in doc1[0]:
                        write.append(str(token).lower())
                        write2.append(token.lemma.lower() + "_" + tags[token.pos])
                        write3.append(token.lemma.lower())
                firstOut.write(" ".join(write)+'\n')
                secondOut.write(" ".join(write2)+'\n')
                thirdOut.write(" ".join(write3)+'\n')

                write = []
                write2 = []
                write3 = []
                for token in doc2[0]:
                        write.append(str(token).lower())
                        write2.append(token.lemma.lower() + "_" + tags[token.pos])
                        write3.append(token.lemma.lower())
                firstOut.write(" ".join(write)+'\n')
                secondOut.write(" ".join(write2)+'\n')
                thirdOut.write(" ".join(write3)+'\n')

                firstOut.write(splitLine[0]+'\n')
                secondOut.write(splitLine[0]+'\n')
                thirdOut.write(splitLine[0]+'\n')

                if i % 100 ==0:
                        print(i,write,write2,write3)
                i+= 1
        firstOut.close()
        secondOut.close()
        thirdOut.close()
