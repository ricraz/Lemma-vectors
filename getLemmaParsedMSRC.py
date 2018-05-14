from pynlp import StanfordCoreNLP

annotators = 'pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

with open('', 'r') as source:
    with open('MSLemmaTagTrain.txt', 'w') as writeFile:
        length = source.readline()
        writeFile.write(length)
        for i in range(int(length)):
            first = source.readline()[:-1]
            second = source.readline()[:-1]
            third = source.readline()
            doc1 = nlp(first)
            doc2 = nlp(second)

            write = []
            for token in doc1[0]:
                write.append(token.lemma.lower() + "_" + tags[token.pos])
            writeFile.write(" ".join(write)+'\n')

            write = []
            for token in doc2[0]:
                write.append(token.lemma.lower() + "_" + tags[token.pos])
            writeFile.write(" ".join(write)+'\n')
            writeFile.write(third)
            if i % 100 == 0:
                print(first, second)
