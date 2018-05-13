from pynlp import StanfordCoreNLP

annotators = 'pos, lemma'

nlp = StanfordCoreNLP(annotators=annotators, options = {'openie.resolve_coref':True})

with ('ppdbLargeFilteredTrain.txt', 'r') as source:
    with ('ppdbLargeFilteredLemmaTagTrain.txt', 'w') as writeFile:
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
