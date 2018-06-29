import re
import os

ppdbFile = open("ppdbLarge.txt", 'r')
outfile = open("ppdbLargeFiltered.txt", 'w')
count = 0
for current in ppdbFile:
    indices = [(a.start(), a.end()) for a in list(re.finditer('\|\|\|',current))[:3]]
    first = current[indices[0][1]+1 : indices[1][0]-1]
    second = current[indices[1][1]+1 : indices[2][0]-1]
    words = set(first.split())
    words.add('-')
    words.add(',')
    words.add('.')
    words.add(':')
    words.add(';')
    words.add('*')
    words.add('-LRB-')
    words.add('-RRB-')
    words2 = set(second.split())
    words2.add('-')
    words2.add(',')
    words2.add('.')
    words2.add(':')
    words2.add(';')
    words2.add('*')
    words2.add('-LRB-')
    words2.add('-RRB-')

    diff = set(words.symmetric_difference(words2))
    if len(diff) >= 6:
        outfile.write(first + os.linesep)
        outfile.write(second + os.linesep)
    if count % 10000 == 0:
        print(count)
        print(first)
        print(second)
        print(diff)
    count += 1
ppdbFile.close()
