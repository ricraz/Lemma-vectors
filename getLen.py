with open("ppdbShuffledTrain.txt", 'r') as ppdb:
    length = int(ppdb.readline())
    best = 0
    for i in range(length):
        ppdb.readline()
        temp = ppdb.readline().split()
        if len(temp) > best:
            best = len(temp)
            print(temp)
        ppdb.readline()
    print(best)
