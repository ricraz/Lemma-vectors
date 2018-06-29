from gensim import models
import numpy as np
from scipy import spatial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#lemmaUntaggedWordVecs = models.KeyedVectors.load_word2vec_format('../../vectors/lemmaUntaggedWordVectors.vec',binary=False).wv

#lemmaUntaggedVecs = models.KeyedVectors.load_word2vec_format('../../vectors/lemmaUntaggedVectors.vec',binary=False)
#lemmaTagVecs = models.KeyedVectors.load_word2vec_format('../../vectors/lemmaTagVectors.vec',binary=False).wv
lemmaTagVecs = models.KeyedVectors.load_word2vec_format('../../vectors/lemmaUntaggedVectors.vec',binary=False).wv

tagMap = {"-v":"_VB", "-n":"_NN", "-j":"_JJ"}
regressor = LinearRegression()
with open('MEN_dataset_lemma_form_full','r') as lemmaFile:
	trueVals = []
	predVals = []
	#lemmaUntaggedVals = []
	#lemmaTagVals = []
	#lemmaTagWordVals = []

	for line in lemmaFile:
		current = line.split()
		trueVals.append(float(current[2]))
		lemma1 = current[0][:-2]
		lemma2 = current[1][:-2]
		#predVals.append([spatial.distance.cosine(lemma1,lemma2)])
		lemmaTag1 = lemma1 + tagMap[current[0][-2:]]
		lemmaTag2 = lemma2 + tagMap[current[1][-2:]]
		
		if lemmaTag1 == "sunglasses_NN":
			lemmaTag1 = "sunglass_NS"
		if lemmaTag2 == "sunglasses_NN":
			lemmaTag2 = "sunglass_NS"
		try:
			vec1 = lemmaTagVecs.wv[lemmaTag1]
		except KeyError:
			lemmaTag1 = lemmaTag1[:-1]+"S"
			try:
				vec1 = lemmaTagVecs.wv[lemmaTag1]
			except KeyError:
				vec1 = np.random.rand(100)
		try:
			vec2 = lemmaTagVecs.wv[lemmaTag2]
		except KeyError:
			lemmaTag2 = lemmaTag2[:-1]+"S"
			try:
				vec2 = lemmaTagVecs.wv[lemmaTag2]
			except KeyError:
				vec2 = np.random.rand(100)
		predVals.append([spatial.distance.cosine(lemmaTagVecs[lemma1], lemmaTagVecs[lemma2])])
	regressor.fit(predVals,trueVals)
with open('MEN_dataset_lemma_form.test','r') as testFile:
	trueVals = []
	predVals = []
	
	for line in testFile:
		current = line.split()
		trueVals.append(float(current[2]))
		lemma1 = current[0][:-2]
		lemma2 = current[1][:-2]
		lemmaTag1 = lemma1 + tagMap[current[0][-2:]]
		lemmaTag2 = lemma2 + tagMap[current[0][-2:]]

		if lemmaTag1 == "sunglasses_NN":
			lemmaTag1 = "sunglass_NS"
		if lemmaTag2 == "sunglasses_NN":
			lemmaTag2 = "sunglass_NS"
		try:
			vec1 = lemmaTagVecs.wv[lemmaTag1]
		except KeyError:
			lemmaTag1 = lemmaTag1[:-1]+"S"
			try:
				vec1 = lemmaTagVecs.wv[lemmaTag1]
			except KeyError:
				vec1 = np.random.rand(100)
		try:
			vec2 = lemmaTagVecs.wv[lemmaTag2]
		except KeyError:
			lemmaTag2 = lemmaTag2[:-1]+"S"
			try:
				vec2 = lemmaTagVecs.wv[lemmaTag2]
			except KeyError:
				vec2 = np.random.rand(100)

		predVals.append([spatial.distance.cosine(lemmaTagVecs[lemma1], lemmaTagVecs[lemma2])])
	c = regressor.predict(predVals)
	print("Loss: ")
	d = mean_squared_error(c,trueVals)
	print(d)
