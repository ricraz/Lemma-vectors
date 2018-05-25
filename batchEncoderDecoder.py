import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import random

import numpy as np

vectorLength = 100
separator = [0]*vectorLength
epochs = 1
inputLength = 100
hiddenLength = 300
batchSize = 1
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

def train(accum, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, lossFunction):
    encoder_hidden = encoder.init_hidden()
    if accum == 0:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    input1_length = input1.size(0)
    input2_length = input2.size(0)
    target_length = target_tensor.size(0)

    for ei in range(input1_length):
        encoder_output, encoder_hidden = encoder(
            input1[ei].view(1,1,-1), encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden

    for di in range(input2_length):
            decoder_output, decoder_hidden = decoder(
                input2[di].view(1,1,-1), decoder_hidden)

    loss = lossFunction(decoder_output, target_tensor)
    
    #loss.backward()
    #if accum == batchSize-1:
    #    encoder_optimizer.step()
    #    decoder_optimizer.step()

    return loss

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
lossFunction = nn.NLLLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

print('starting training')

with open('ppdbLargeFilteredLemmaTagTrain.txt','r') as ppdb:
	fileSize = int(ppdb.readline())
	traininSet = []
	for i in range(fileSize):
		trainingSet.append([])
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(int(ppdb.readline()))
	
	validationSize = fileSize//10
	trainingSize = fileSize - validationSize
	trainingSet, validationSet = trainingSet[:trainingSize],trainingSet[trainingSize:]

	for i in range(epochs):
		avg_loss = 0.0
		current_loss = 0.0
		losses = []
		for j in range(trainingSize):
			input1 = Variable(torch.Tensor(trainingSet[j][0]))
			input2 = Variable(torch.Tensor(trainingSet[j][1]))

			target_tensor = Variable(torch.LongTensor([trainingSize[j][2]]))
			loss = train(j%batchSize, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, lossFunction)

			avg_loss += loss.data[0]
			current_loss += loss.data[0]
			losses.append(loss)

			if j%batchSize==batchSize-1:
				batchLoss = sum(losses)/batchSize
				batchLoss.backward()
				encoder_optimizer.step()
				decoder_optimizer.step()
				losses = []
				if j % 500 == 1:
					print('epoch : ', i, ', batch : ',j//batchSize, ', loss : ', current_loss/500,', overall: ', avg_loss/j)
					current_loss = 0
		random.shuffle(allLines)

def testModel(
		avg_loss = 0.0
		true_Positives = 0
		false_Positives = 0
		false_Negatives = 0
		true_Negatives = 0
		with open('ppdbLargeFilteredLemmaTagTest.txt','r') as ppdb:
			testFileSize = int(ppdb.readline())
			for j in range(testFileSize):
				data1 = getVector(ppdb.readline()[:-1],vectors)
				data2 = getVector(ppdb.readline()[:-1],vectors)
				input1 = Variable(torch.Tensor(data1))
				input2 = Variable(torch.Tensor(data2))
				target = int(ppdb.readline())
				target_tensor = Variable(torch.LongTensor([target]))
				output = evaluate(encoder, decoder, input1, input2)
				loss = lossFunction(output, target_tensor)

				avg_loss += loss.data[0]
		
				bigger = (output[0][1] > output[0][0]).data.numpy()
				if target==1:
					if bigger:
						true_Positives += 1
					else:
						false_Negatives += 1

				else:
					if bigger:
						false_Positives += 1
					else:
						true_Negatives += 1

		precision = true_Positives/(true_Positives+false_Positives)
		recall = true_Positives/(true_Positives+false_Negatives)
		f1 = 2 * precision*recall/(precision+recall)
		print('Accuracy: ', (true_Positives + true_Negatives)/fileSize)
		print('Precision: ', precision)
		print('Recall: ', recall)
		print('F1: ', f1)
