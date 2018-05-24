import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

import numpy as np

vectorLength = 100
separator = [0]*vectorLength
epochs = 1
inputLength = 100
hiddenLength = 300
batchSize = 128
randomSeed = 0

with open('../fastText/lemmaTagSkipWordVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

"""with open('../fastText/missingLemmaUntaggedSkipVectors.txt', 'r') as missingFile:
	for line in missingFile:
		splitLine = line.split()
		start = len(splitLine) - vectorLength
		vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)"""

def getVector(line, vectors):
        output = []
        split = line.split()
        for word in split:
                if word in vectors:
                        output.append(vectors[word])                        
                else:
                        print("Problem", word)
                        output.append(np.random.uniform(-1, 1, vectorLength)) #accounting for unknown vectors
        return output

class EncoderLSTM(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
                super(EncoderLSTM,self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        def forward(self, inputs, hidden):
                lstm_out, lstm_h = self.lstm(inputs, hidden)
                return lstm_out, lstm_h
        def init_hidden(self):
                return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))

class DecoderLSTM(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
                super(DecoderLSTM,self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
                self.linearOut = nn.Linear(hidden_dim,2)
                self.softmax = nn.LogSoftmax()
        def forward(self, inputs, hidden):
                lstm_out, lstm_h = self.lstm(inputs, hidden)
                x = lstm_out[-1]
                x = self.linearOut(x)
                x = F.log_softmax(x)
                return x, lstm_h

def train(inputs, batchSize, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function):
    
    zipped = zip(*inputs)
    input1 = Variable(torch.Tensor(next(zipped))
    for 
    input2 = Variable(torch.Tensor(next(zipped))
    target_tensor = Variable(torch.LongTensor(next(zipped)))

    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input1_length = input1.size(0)
    input2_length = input2.size(0)
    target_length = target_tensor.size(0)

    for ei in range(input1_length):
        encoder_output, encoder_hidden = encoder(
            input1[ei].view(1,batchSize,-1), encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden

    for di in range(input2_length):
            decoder_output, decoder_hidden = decoder(
                input2[di].view(1,batchSize,-1), decoder_hidden)
            
    loss = loss_function(decoder_output, target_tensor)
    
    loss.backward()

    decoder_optimizer.step()
    encoder_optimizer.step()

    return loss.data[0]

def evaluate(encoder, decoder, input1, input2):
	input1_length = input1.size()[0]
	input2_length = input2.size()[0]
	encoder_hidden = encoder.init_hidden()

	for ei in range(input1_length):
		encoder_output, encoder_hidden = encoder(input1[ei].view(1,1,-1), encoder_hidden)
	
	decoder_hidden = encoder_hidden
	
	for di in range(input2_length):
		decoder_output, decoder_hidden = decoder(input2[di].view(1,1,-1), decoder_hidden)
	return decoder_output

encoder = EncoderLSTM(inputLength, hiddenLength)
decoder = DecoderLSTM(inputLength, hiddenLength)
loss_function = nn.NLLLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

print('starting training')

with open('ppdbLargeFilteredLemmaTagTrain.txt','r') as ppdb:
	fileSize = int(ppdb.readLine())
	allLines = []
	for i in range(fileSize):
		allLines.append([])
		allLines[-1].append(getVector(ppdb.readline()[:-1],vectors))
		allLines[-1].append(getVector(ppdb.readline()[:-1],vectors))
		allLines[-1].append(int(ppdb.readline()))

	for i in range(epochs):
		avg_loss = 0.0
		j = 0
		while (j+1)*batchSize <= fileSize:
			loss = train(allLines[j*batchSize, batchSize, (j+1)*batchSize], encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function)
			avg_loss += loss	
			print('epoch: " ', i, ' batch: ', j, ' loss: ', loss, ' overall: ', str(avg_loss/(j*batchSize)))
			j += 1

		if j*batchSize < fileSize:
			loss = train(allLines[j*batchSize:], fileSize - j*batchSize, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function)
			avg_loss += loss
		print('the average loss after completion of %d epochs is %g'%(j, avg_loss/fileSize))

avg_loss = 0.0
success = 0
with open('ppdbLargeFilteredLemmaTagTest.txt','r') as ppdb:
	fileSize = int(ppdb.readline())
	for j in range(fileSize):
		data1 = getVector(ppdb.readline()[:-1],vectors)
		data2 = getVector(ppdb.readline()[:-1],vectors)
		input1 = Variable(torch.Tensor(data1))
		input2 = Variable(torch.Tensor(data2))
		target = int(ppdb.readline())
		target_tensor = Variable(torch.LongTensor([target]))
		output = evaluate(encoder, decoder, input1, input2)
		loss = loss_function(output, target_tensor)

		avg_loss += loss.data[0]
		
		bigger = (output[0][1] > output[0][0]).data.numpy()
		if (target == 1 and bigger) or (target==0 and not bigger):
			success += 1
print("Success: " + str(success/fileSize))
print(str(avg_loss/fileSize))

#with open('lemma_untagged_shuffled_model_save', 'wb') as f:
#	torch.save(model,f)

#print("saved")