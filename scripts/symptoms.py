import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import tflearn
import tensorflow as tf
import random


def what_are_your_symptoms():
	print("What are your symptoms (please list with commas):")
	patient_symptoms = input().split(",")
	print(patient_symptoms)
	potential_diagnosis = []
	with open('disease.json','r') as f:
		diseases = json.load(f)
		for patient_symptom in patient_symptoms:
			for disease in diseases:
				if patient_symptom in diseases[disease]['symptoms']:
					potential_diagnosis.append(diseases[disease]['disease'])
		print(potential_diagnosis)
				

#what_are_your_symptoms()

with open('intents.json') as json_data:
	intents = json.load(json_data)

nltk.download('punkt')

words = []
classes = []
documents = []

for intent in intents['intents']:
	for pattern in intent['patterns']:
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		documents.append((w,intent['tag']))
		if intent['tag'] not in classes:
			classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])




