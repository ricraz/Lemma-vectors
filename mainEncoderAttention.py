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
MAX_LENGTH = 10

with open('../fastText/unlemmaUntaggedPuncSkipVectors.vec', 'r') as vectorFile:
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
                print(inputs.size())
                print(hidden[0].size())
                print(hidden[1].size())
                print("Threee down")
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
            print(output.size())
            print(hidden[0].size())
            print(hidden[1].size())
            print("three more")
            output, hidden = self.lstm(output, hidden)
            print("LSTM WORKED!")
            output = F.log_softmax(self.linearOut(output[-1]))
            return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_dim)

def train(input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length = MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

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
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss

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
loss_function = nn.NLLLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

print('starting training')

for i in range(epochs) :
	avg_loss = 0.0
	last_avg = 0.0
	with open('ppdbLargeFilteredTrain.txt','r') as ppdb:
		fileSize = int(ppdb.readline())
		for j in range(fileSize):
			data1 = getVector(ppdb.readline()[:-1], vectors)
			data2 = getVector(ppdb.readline()[:-1], vectors)
			#print(data1+[seperator]+data2)
			input1 = Variable(torch.Tensor(data1))
			input2 = Variable(torch.Tensor(data2))
			
			target = int(ppdb.readline())
			target_tensor = Variable(torch.LongTensor([target]))

			loss = train(input1, input2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function)
			#print(loss)
			avg_loss += loss.data[0]
			if j%500 == 1:
				print('epoch : ', i, ' iterations : ',j, 'loss : ', loss.data[0],'; overall: ', str((avg_loss-last_avg)/500))
				last_avg = avg_loss

#	torch.save(model.state_dict(), 'model1' + str(i+1)+'.pth')			
	print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/fileSize)))	

avg_loss = 0.0
success = 0
with open('ppdbLargeFilteredTest.txt','r') as ppdb:
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
