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
global maxLength
maxLength = 6

with open('fastText/lemmaVectors.vec', 'r') as file:
        vectors = dict()
        file.readline()
        for line in file:
                splitLine = line.split()
                start = len(splitLine) - vectorLength
                try:
                        vectors[" ".join(splitline[0:start])] = np.array(splitLine[start:]).astype(float)
                except ValueError:
                        print(splitLine, len(splitLine))

max_sequence_len = 10

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

class Model(torch.nn.Module) :
	def __init__(self,embedding_dim,hidden_dim) :
		super(Model,self).__init__()
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		self.linearOut = nn.Linear(hidden_dim,2)
	def forward(self,inputs,hidden) :
		x = self.embeddings(inputs).view(len(inputs),1,-1)
		lstm_out,lstm_h = self.lstm(x,hidden)
		x = lstm_out[-1]
		x = self.linearOut(x)
		x = F.log_softmax(x)
		return x,lstm_h
	def init_hidden(self) :
		return (Variable(torch.zeros(1, 1, self.hidden_dim)),Variable(torch.zeros(1, 1, self.hidden_dim)))	

model = Model(100,100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 4

torch.save(model.state_dict(), 'model' + str(0)+'.pth')
print('starting training')

sep = [0]*100

with open("ppdbLargeFilteredTrain.txt", 'r') as ppdb:
        for i in range(epochs):
                avg_loss = 0.0
        
                for j in range(len(ppdb)//3):
                        first = getVector(ppdb.readline()[:-1], vectors)
                        second = getVector(ppdb.readline()[:-1], vectors)
                        third = int(ppdb.readline())

                        input_data = Variable(torch.Tensor(first+[sep]+second)
                        hidden = model.init_hidden()
                        y_pred,_ = model(input_data, hidden)
                        model.zero_grad()
                        loss = loss_function(y_pred, torch.Tensor([1-third, third]))
                        avg_loss += loss.data[0]

                        if j%500 == 0:
                                print('epoch :%d iterations :%d loss :%g'%(i, j,loss.data[0]))
                        loss.backward()
                        optimizer.step()

                torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')			
                print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/len(data))))
                
"""	for idx,lines in enumerate(f) :
		if not idx == 0 :
			data = lines.split('\t')[2]
			data = normalizeString(data).strip()
			input_data = [obj1.word_to_idx[word] for word in data.split(' ')]
			if len(input_data) > max_sequence_len :
				input_data = input_data[0:max_sequence_len]

			input_data = Variable(torch.LongTensor(input_data))
			target = int(lines.split('\t')[1])
			target_data = Variable(torch.LongTensor([target]))
			hidden = model.init_hidden()
			y_pred,_ = model(input_data,hidden)
			model.zero_grad()
			loss = loss_function(y_pred,target_data)
			avg_loss += loss.data[0]
			
			if idx%500 == 0 or idx == 1:
				print('epoch :%d iterations :%d loss :%g'%(i,idx,loss.data[0]))
				
			loss.backward()
			optimizer.step()
	torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')			
	print('the average loss after completion of %d epochs is %g'%((i+1),(avg_loss/len(f))))	
"""

with open('dict.pkl','wb') as f :
	pickle.dump(obj1.word_to_idx,f)

