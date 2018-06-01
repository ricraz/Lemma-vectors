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
epochs = 50
inputLength = 100
hiddenLength = 300
MAX_LENGTH = 6
batchSize - 64
randomSeed = 0
torch.manual_seed(randomSeed)
resultsFile = open('results/...', 'w')
resultsFile.write('Random seed: ' + str(randomSeed) + '\n')
resultsFile.write('BatchSize: ' + str(batchSize) + '\n')

with open('../fastText/unlemmaUntaggedPuncSkipVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                #for i in range(start, len(splitLine)):
                #        splitLine[i] = float(splitLine[i])
                vectors[" ".join(splitLine[0:start])] = np.array(splitLine[start:], dtype = float)

with open('../fastText/missingLemmaUntaggedSkipVectors.txt', 'r') as missingFile:
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
                        raise(ValueError)
                        #output.append(np.random.uniform(-1, 1, vectorLength)) #accounting for unknown vectors
        return output

class EncoderLSTM(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
                super(EncoderLSTM,self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        def forward(self, inputs, hidden):
                #print(inputs.size())
                #print(hidden[0].size())
                #print(hidden[1].size())
                #print("Threee down")
                lstm_out, lstm_h = self.lstm(inputs.view(1,1,-1), hidden)
                return lstm_out, lstm_h
        def init_hidden(self):
                return (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))

class AttentionDecoderLSTM(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_p=0.1, max_length = MAX_LENGTH):
            super(AttentionDecoderLSTM,self).__init__()
            self.hidden_dim = hidden_dim
            self.dropout_p = dropout_p
            self.max_length = max_length

            self.attention = nn.Linear(hidden_dim+embedding_dim, max_length)
            self.attention_combine = nn.Linear(hidden_dim+embedding_dim, self.hidden_dim)
            self.dropout = nn.Dropout(self.dropout_p)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)
            self.linearOut = nn.Linear(hidden_dim, 2)
            self.softmax = nn.LogSoftmax()

    def forward(self,inputs,hidden,encoder_outputs):
            embedded = inputs.view(1,1,-1)
            attention_weights = F.softmax(self.attention(torch.cat((embedded,hidden[0]),dim=2)))
            encoderThing = Variable(encoder_outputs.view(1,MAX_LENGTH,-1))
            attention_applied = torch.bmm(attention_weights, encoderThing)
            output = torch.cat((embedded, attention_applied[0].unsqueeze(0)),dim=2)
            output = self.attention_combine(output)
            output = F.relu(output)
            #print(output.size())
            #print(hidden[0].size())
            #print(hidden[1].size())
            #print("three more")
            output, hidden = self.lstm(output, hidden)
            print("LSTM WORKED!")
            output = F.log_softmax(self.linearOut(output[-1]))
            return output, hidden, attention_weights

    #def init_hidden(self):
    #    return torch.zeros(1,1,self.hidden_dim)

def train(accum, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length = MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    input1_length = input1.size(0)
    input2_length = input2.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim)

    for ei in range(input1_length):
        encoder_output, encoder_hidden = encoder(
            input1[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output.data[0, 0]

    decoder_hidden = encoder_hidden

    for di in range(input2_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                input2[di], decoder_hidden, encoder_outputs)
            print(di)
            
    loss = loss_function(decoder_output, target_tensor)
    
    #loss.backward()

    #encoder_optimizer.step()
    #decoder_optimizer.step()

    return loss

    #encoder_optimizer.zero_grad()
    #decoder_optimizer.zero_grad()

def evaluate(encoder, decoder, input1, input2, max_length = MAX_LENGTH):
	input1_length = input1.size()[0]
	input2_length = input2.size()[0]
	encoder_hidden = encoder.init_hidden()

	encoder_outputs = torch.zeros(max_length, encoder.hidden_dim)

	for ei in range(input1_length):
		encoder_output, encoder_hidden = encoder(input1[ei], encoder_hidden)
		encoder_outputs[ei] += encoder_output[0,0]
	
	decoder_hidden = encoder_hidden
	decoder_attentions = torch.zeros(max_length, max_length)
	
	for di in range(input2_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(
                    input2[di], decoder_hidden, encoder_outputs)
		decoder_attentions[di] = decoder_attention.data

	return decoder_output

encoder = EncoderLSTM(inputLength, hiddenLength)
decoder = AttentionDecoderLSTM(inputLength, hiddenLength)
loss_function = nn.NLLLoss(size_average=False)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

print('starting training')

def testModel(dataFile, encoder, decoder, loss_function):
	avg_loss = 0.0
	true_Positives = 0
	false_Positives = 0
	false_Negatives = 0
	true_Negatives = 0
	fileSize = len(dataFile)
	for j in range(fileSize):
		input1 = Variable(torch.Tensor(dataFile[j][0]))
		input2 = Variable(torch.Tensor(dataFile[j][1]))
		target = dataFile[j][2]
		target_tensor = Variable(torch.LongTensor([target]))
		output = evaluate(encoder, decoder, input1, input2)
		loss = loss_function(output, target_tensor)
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
	resultsFile.write('Average loss: ' + str(avg_loss/fileSize) + '\n')
	resultsFile.write('Accuracy: ' + str((true_Positives + true_Negatives)/fileSize) + '\n')
	#resultsFile.write('Precision: ', precision)
        #resultsFile.write('Recall: ', recall)
	resultsFile.write('F1: ' + str(f1) + '\n') 
	return f1

with open('ppdbShuffledLemmaTagTrain.txt','r') as ppdb:
	fileSize = int(ppdb.readline())
	trainingSet = []
	for i in range(fileSize):
		trainingSet.append([])
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(getVector(ppdb.readline()[:-1],vectors))
		trainingSet[-1].append(int(ppdb.readline()))
	
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

			target_tensor = Variable(torch.LongTensor([trainingSet[j][2]]))
			loss = train(j%batchSize, input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function)

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
				print('epoch : ', i, ', batch : ',j//batchSize, ', loss : ', current_loss/500,', overall: ', avg_loss/j)
				current_loss = 0

		random.shuffle(trainingSet)
		resultsFile.write('epoch ' + str(i+1) + ' validation scores: \n')
		currentValidation = testModel(validationSet, encoder, decoder, loss_function)
		validations.append(currentValidation)

		with open('ppdbShuffledLemmaTagTest.txt','r') as ppdbTest:
			testFileSize = int(ppdbTest.readline())
			testSet = []
			for k in range(testFileSize):
				testSet.append([])
				testSet[-1].append(getVector(ppdbTest.readline()[:-1],vectors))
				testSet[-1].append(getVector(ppdbTest.readline()[:-1],vectors))
				testSet[-1].append(int(ppdbTest.readline()))
			resultsFile.write('epoch ' + str(i+1) + ' test scores: \n')
			testScore = testModel(testSet, encoder, decoder, loss_function)
			testScores.append(testScore)
                resultsFile.write(str(validations)+'\n')
        	resultsFile.write(str(testScores)+'\n')
resultsFile.close()

