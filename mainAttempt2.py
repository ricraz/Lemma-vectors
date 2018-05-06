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
seperator = [0]*vectorLength


with open('fastText/lemmaVectors.vec', 'r') as vectorFile:
        vectors = dict()
        vectorFile.readline()
        for line in vectorFile:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                try:
                        vectors[" ".join(splitLine[0:start])] = splitLine[start:]
                except ValueError:
                        print(splitLine, len(splitLine))


def getVector(line, vectors):
        output = []
        split = line.split()
        for word in split:
                tempWord = lemmatizer.lemmatize(word.lower())
                if tempWord in vectors:
                        output.append(vectors[tempWord])
                else:
                        output.append(list(np.random.uniform(-1, 1, vectorLength))) #accounting for unknown vectors
        return output

class Model(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
                super(Model,self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
                self.linearOut = nn.Linear(hidden_dim,2)
        def forward(self, inputs, hidden):
                x = torch.Tensor(inputs).view(len(inputs),1,-1)
                lstm_out, lstm_h = self.lstm(x, hidden)
                x = lstm_out[-1]
                x = self.linearOut(x)
                x = F.softmax(x)
                return x, lstm_h
        def init_hidden(self):
                return (Variable(torch.zeros(1,1,self.hidden_dim)), Variable(torch.zeros(1,1,self.hidden_dim)))

model = Model(100, 100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 1

torch.save(model.state_dict(), 'model' + str(0)+'.pth')
print('starting training')

for i in range(epochs) :
	avg_loss = 0.0
	with open('ppdbLargeFilteredTrain.txt','r') as ppdb:
                fileSize = int(ppdb.readline())
                for j in range(fileSize):
			data1 = getVector(ppdb.readline()[:-1], vectors)
			data2 = getVector(ppdb.readline()[:-1], vectors)
			input_data = Variable(torch.LongTensor(data1 + [seperator] + data2))
			
			target = int(ppdb.readline())
			target_data = Variable(torch.LongTensor([target]))
			hidden = model.init_hidden()
			y_pred,_ = model(input_data,hidden)
			model.zero_grad()
			loss = loss_function(y_pred,target_data)
			avg_loss += loss.data[0]
			
			if j%500 == 0 or j == 1:
				print('epoch :%d iterations :%d loss :%g'%(i,j,loss.data[0]))

				
			loss.backward()
			optimizer.step()
	torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')			
	print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/len(f))))	

with open('dict.pkl','wb') as f :
	pickle.dump(obj1.word_to_idx,f)
