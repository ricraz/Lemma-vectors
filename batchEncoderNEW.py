import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
from scipy.stats import spearmanr
import numpy as np

vectorLength = 100
epochs = 50
inputLength = 100
hiddenLength = 300



batchSize = 64
randomSeed = 0
torch.manual_seed(randomSeed)
resultsFile = open('results2/unlemmaTaggedWord.txt', 'w')
resultsFile.write('Random seed: ' + str(randomSeed) + '\n')
resultsFile.write('BatchSize: ' + str(batchSize) + '\n')

with open('../vectors/unlemmaTagWordVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

with open('missing/missingUNLEMMATAGGEDVectors.txt', 'r') as missingFile:
        for line in missingFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                if " ".join(splitLine[0:start]) not in vectors:
                        vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

def getVector(line, vectors):
        output = []
        split = line.split()
        for word in split:
                if word in vectors:
                        output.append(vectors[word])
                elif word[0] == '(':
                        output.append(vectors['-lrb-'+word[1:]])
                elif word[0] == ')':
                        output.append(vectors['-rrb-'+word[1:]])
                else:
                        print("Problem", word)
                        print(line)
                        #print(tot)
                        #tot += 1
                        raise(ValueError)
                        #output.append(np.random.uniform(-1, 1, vectorLength)) #accounting for unknown vectors
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
                self.linearOut = nn.Linear(hidden_dim,1)
                #self.softmax = nn.LogSoft
                self.sigmoidOut = nn.Sigmoid()

        def forward(self, inputs, hidden):
                lstm_out, lstm_h = self.lstm(inputs, hidden)
                x = lstm_out[-1]
                x = self.linearOut(x)
                x = self.sigmoidOut(x)
                return x, lstm_h

def train(accum, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, lossFunction):
    encoder_hidden = encoder.init_hidden()

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
    loss = lossFunction(decoder_output, target_tensor.view(1,1))
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

def testModel(dataFile, encoder, decoder, lossFunction):
	avg_loss = 0.0
	fileSize = len(dataFile)
	spearman = []
	targetVec = []

	for j in range(fileSize):
		input1 = Variable(torch.Tensor(dataFile[j][0]))
		input2 = Variable(torch.Tensor(dataFile[j][1]))
		target = dataFile[j][2]
		target_tensor = Variable(torch.Tensor([target]))
		output = evaluate(encoder, decoder, input1, input2)
		spearman.append(output.item())
		targetVec.append(target)

		loss = lossFunction(output, target_tensor.view(1,1))
		avg_loss += loss.item()

	spearmansR = spearmanr(spearman, targetVec)
	print(spearmansR)
	resultsFile.write('Average loss: ' + str(avg_loss/fileSize) + '\n')
	resultsFile.write('Spearman: ' + str(spearmansR[0]) + '\n')
	#resultsFile.write('Precision: ', precision)
        #resultsFile.write('Recall: ', recall)
	return spearmansR[0]

encoder = EncoderLSTM(inputLength, hiddenLength)
decoder = DecoderLSTM(inputLength, hiddenLength)
lossFunction = nn.MSELoss(size_average=False)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

print('starting training')

with open('unlemmaTagPPDBTrain.txt','r') as ppdb:
	fileSize = int(ppdb.readline())
	trainingSet = []
	for i in range(fileSize):
		trainingSet.append([])
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(((float(ppdb.readline())-1)/4))
	
	validationSize = fileSize//10
	trainingSize = fileSize - validationSize
	trainingSet, validationSet = trainingSet[:trainingSize],trainingSet[trainingSize:]

	validations = []
	testScores = []
        
	for i in range(epochs):
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()
		avg_loss = 0.0
		current_loss = 0.0
		losses = []
		for j in range(trainingSize):
			input1 = Variable(torch.Tensor(trainingSet[j][0]))
			input2 = Variable(torch.Tensor(trainingSet[j][1]))

			target_tensor = torch.Tensor([trainingSet[j][2]])
			loss = train(j%batchSize, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, lossFunction)

			avg_loss += loss.data[0]
			current_loss += loss.data[0]
			losses.append(loss)

			if j%batchSize==batchSize-1:
				batchLoss = sum(losses)/batchSize
				batchLoss.backward()
				encoder_optimizer.step()
				decoder_optimizer.step()
				encoder_optimizer.zero_grad()
				decoder_optimizer.zero_grad()

				losses = []
				print('epoch : ', i, ', batch : ',j//batchSize, ', loss : ', current_loss/batchSize,', overall: ', avg_loss/j)
				current_loss = 0

		random.shuffle(trainingSet)
		resultsFile.write('epoch ' + str(i+1) + ' validation scores: \n')
		currentValidation = testModel(validationSet, encoder, decoder, lossFunction)
		validations.append(currentValidation)

		if True:
			with open('unlemmaTagPPDBTest.txt','r') as ppdbTest:
				testFileSize = int(ppdbTest.readline())
				testSet = []
				for k in range(testFileSize):
					testSet.append([])
					testSet[-1].append(getVector(ppdbTest.readline()[:-1],vectors))
					testSet[-1].append(getVector(ppdbTest.readline()[:-1],vectors))
					testSet[-1].append(((float(ppdbTest.readline())-1)/4))
				resultsFile.write('epoch ' + str(i+1) + ' test scores: \n')
				testScore = testModel(testSet, encoder, decoder, lossFunction)
				testScores.append(testScore)
		resultsFile.write(str(validations)+'\n')
		resultsFile.write(str(testScores)+'\n')
resultsFile.close()
