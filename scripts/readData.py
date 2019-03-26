import nltk
import numpy as np
import random
import string
import json
from flask import Flask, render_template, request


f=open('medtext.txt','r',errors = 'ignore')

raw=f.read()

raw=raw.lower()# converts to lowercase

nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

print(sent_tokens[:2])
print(word_tokens[:2])

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

def patientIntake():
	print("MAIA: Hello, welcome to the MAIA clinic.  I am here to help with patient intake.  First tell me your name.")
	name = input()
	print("MAIA: Street Address?")
	streetAddr = input()
	print("MAIA: City?")
	city = input()
	print("MAIA: Phone Number?")
	number = input()
	print("MAIA: Email?")
	email = input()
	print("MAIA: What way would you prefer to be contacted?")
	contactPreference = input()
	print("MAIA: What is your date of birth?")
	dOB = input()
	print("MAIA: Age?")
	age = input()
	print("MAIA: Gender?")
	gender = input()
	print("MAIA: Are you on any medications at the moment?")
	medications = input()
	print("MAIA: What brings you into the clinic today?")
	symptoms = input()
	print("MAIA: Thankyou, your information will be passed to a your doctor and we will see you shortly.")
	with open("patientInfo.json") as f:
		data = json.load(f)
		patientRecord = {}
		patientRecord["Name"]=name
		patientRecord["Address"]=streetAddr
		patientRecord["City"]=city
		patientRecord["Phone Number"]=number
		patientRecord["Email"]=email
		patientRecord["ContactPref"]=contactPreference
		patientRecord["Birthdate"]=dOB
		patientRecord["Age"]=age
		patientRecord["Gender"]=gender
		patientRecord["Medications"]=medications
		patientRecord["Symptoms"]=symptoms
		data["PatientRecords"].append(patientRecord)
	with open("patientInfo.json",'w') as fs:
		json.dump(data,fs)

flag=True
print("MAIA: My name is Maia. Are you a doctor or a patient? If you want to exit, type Bye!")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response=='patient'):
    	patientIntake()
    	flag=False
    elif(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("MAIA: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("MAIA: "+greeting(user_response))
            else:
                print("MAIA: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("MAIA: Bye! take care..")

























