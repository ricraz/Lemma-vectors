import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle

import numpy as np

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

vectorLength = 100
separator = [0]*vectorLength


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

class Model(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
                super(Model,self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
                self.linearOut = nn.Linear(hidden_dim,2)
        def forward(self, inputs, hidden):
                #x = torch.Tensor(inputs).
                lstm_out, lstm_h = self.lstm(inputs, hidden)
                x = lstm_out[-1]
                x = self.linearOut(x)
                x = F.log_softmax(x)
                return x, lstm_h
        def init_hidden(self):
                return (Variable(torch.zeros(1,1,self.hidden_dim)), Variable(torch.zeros(1,1,self.hidden_dim)))

model = Model(100, 300)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 1

torch.save(model.state_dict(), 'model1' + str(0)+'.pth')
print('starting training')

for i in range(epochs) :
	avg_loss = 0.0
	last_avg = 0.0
	with open('ppdbLargeFilteredLemmaTagTrain.txt','r') as ppdb:
		fileSize = int(ppdb.readline())
		for j in range(fileSize):
			data1 = getVector(ppdb.readline()[:-1], vectors)
			data2 = getVector(ppdb.readline()[:-1], vectors)
			#print(data1+[seperator]+data2)
			input_data = Variable(torch.Tensor(data1 + [separator] + data2))
			
			target = int(ppdb.readline())
			target_data = Variable(torch.LongTensor([target]))
			hidden = model.init_hidden()
			y_pred,_ = model(input_data,hidden)
			model.zero_grad()
			loss = loss_function(y_pred,target_data)
			avg_loss += loss.data[0]
			if j%500 == 1:
				print('epoch : ', i, ' iterations : ',j, 'loss : ', loss.data[0],'; overall: ', str((avg_loss-last_avg)/500))
				last_avg = avg_loss

			loss.backward()
			optimizer.step()
	torch.save(model.state_dict(), 'model1' + str(i+1)+'.pth')			
	print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/fileSize)))	

avg_loss = 0.0
success = 0
with open('ppdbLargeFilteredLemmaTagTest.txt','r') as ppdb:
	fileSize = int(ppdb.readline())
	for j in range(fileSize):
		data1 = getVector(ppdb.readline()[:-1],vectors)
		data2 = getVector(ppdb.readline()[:-1],vectors)
		input_data = Variable(torch.Tensor(data1 + [separator] + data2))
		target = int(ppdb.readline())
		target_data = Variable(torch.LongTensor([target]))
		hidden = model.init_hidden()
		y_pred,_ = model(input_data,hidden)
		model.zero_grad()
		loss = loss_function(y_pred,target_data)
		avg_loss += loss.data[0]
		
		bigger = (y_pred[0][1] > y_pred[0][0]).data.numpy()
		if (target == 1 and bigger) or (target==0 and not bigger):
			success += 1
print(str(success/fileSize))
print(str(avg_loss/fileSize))

with open('lemma_untagged_shuffled_model_save', 'wb') as f:
	torch.save(model,f)

print("saved")
