import msgpack
import gzip
import string

fin = gzip.open('wikipedia.msgpack.gz', 'rb')
unpacker = msgpack.Unpacker(fin, encoding='utf-8')
puncTags = set(['.',':','#',"''",',','``','$','-RRB-','-LRB-'])
punc = set(string.punctuation)
punc.remove('-')
sentences = open("sentencesLemmatisedWithPunc.txt",'w')

tags = {'PDT': 'PT', '.': '.', 'VB': 'VB', ':': ':', '#': '#', 'VBN': 'VN', 'RBR': 'RR', 'PRP$': 'PR', 'DT': 'DT', 'VBZ': 'VZ', 'CC': 'CC', 'TO': 'TO', 'LS': 'LS', 'SYM': 'SM', 'RBS': 'RS', 'JJ': 'JJ', 'EX': 'EX', 'WP': 'WP', 'POS': 'PS', 'WDT': 'WT', 'VBP': 'VP', 'WRB': 'WB', 'PRP': 'PP', 'JJR': 'JR', 'VBD': 'VD', 'NNPS': 'NQ', 'RB': 'RB', '-LRB-': 'L$', 'RP': 'RP', 'JJS': 'JS', 'CD': 'CD', '-RRB-': 'R$', 'NNP': 'NP', '$': '$', 'WP$': 'WP', 'FW': 'FW', 'VBG': 'VG', "''": "''", ',': ',', 'NN': 'NN', 'UH': 'UH', 'NNS': 'NS', 'MD': 'MD', '``': '``', 'IN': 'IN'}

for i, sent in enumerate(unpacker):
	sentences.write(" ".join(sent[1]).lower() + '\n')
	if i % 100000 == 0:
		print(" ".join(sent[1]).lower())
"""
for i, sent in enumerate(unpacker):
    stripped = [sent[1][i].lower() for i in range(len(sent[1]))] if sent[2][i] not in puncTags]
    if stripped != []:
        sentences.write(" ".join(stripped)+'\n')
    if i %100000 == 0:
        print(i)
        print(" ".join(stripped)+'\n')
"""
fin.close()
sentences.close()
#print(count)
print(tags)
#print(count)
#model = fasttext.skipgram('sentences.txt', 'model')
#print(model.words)
#model = fasttext.cbow('sentences.txt', 'model')
#print(model.words)"""
