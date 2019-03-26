import nltk
import numpy as np
import random
import string
import json
from flask import Flask, render_template, request
from search_biopython import search, fetch_details
import os


f=open('../rawdata/medtext.txt','r',errors = 'ignore')

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



def patientIntake(index):
	switch = {
		0:["Phone Number","Hello, welcome to the MAIA clinic.  I am here to help with patient intake.  First tell me your phone number."],
		1:["Street Address" ,"And what is your street address?"],
		2:["City", "What city is that in?"],
		3:["Email", "What is your date of birth?"],
		4:["Gender", "And what is your gender?"],
		5:["Symptoms","Could you tell me some of your symptoms?"],
		6:["Medications", "Are you on any medications at the moment?"],
		7:["Perfect, now that we know a bit about you do you have any questions for me?"]
	}

	return switch[index]
	

app = Flask(__name__)

with open('../msgHistory/msg.json','w') as f:
	MESSAGES = {}
	MESSAGES["messages"]=[]
	botGreeting = {}
	botGreeting["sender"]="bot"
	botGreeting["message"]=greeting("Hello")
	MESSAGES["messages"].append(botGreeting)
	json.dump(MESSAGES,f)

with open('../msgHistory/msg0.json','w') as f:
	MESSAGES = {}
	MESSAGES["messages"]=[]
	botGreeting = {}
	botGreeting["sender"]="bot"
	botGreeting["message"]=greeting("Hello")
	MESSAGES["messages"].append(botGreeting)
	json.dump(MESSAGES,f)


def newPatient(name):
	print("new patient")
	patientfile = '../msgHistory/patients/'+name+'.json'
	patientinfo = {}
	patientinfo["name"]=name
	patientinfo["PATIENTINDEX"]=0
	patientinfo["messages"]=[]
	with open(patientfile,'w+') as f:
		json.dump(patientinfo,f)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/patientQ', methods=['POST'])
def patientQ():
	return render_template('patientQ.html')


@app.route('/doctor', methods=['POST'])
def doctor():
	# msgList = os.listdir('../msgHistory/')
	# highMsg = msgList[len(msgList)]
	# newMsg = hignMsg[-6]+1
	# print(newMsg)
	# FILENAME = 'msg'+str(newMsg)+'.json'
	# print(FILENAME)
	# with open('../msgHistory/'+FILENAME, 'w') as f:
	# 	json.dump(MESSAGES, f)
	return render_template('doctor.html',MESSAGES=MESSAGES)

@app.route('/patient', methods=['POST'])
def patient():

	patient_name=request.form['patient_name'].lower().replace(' ','')
	print(patient_name)
	patients = os.listdir('../msgHistory/patients/')
	if patient_name+'.json' not in patients:
		newPatient(patient_name)
		with open('../msgHistory/patients/'+patient_name+'.json') as f:
			history=json.load(f)
		bot_response = patientIntake(history["PATIENTINDEX"])[1]
	else:
		with open('../msgHistory/patients/'+patient_name+'.json') as f:
			history=json.load(f)
		bot_response = greeting('Hi')

	messages=history["messages"]

	botBlock={}
	botBlock["message"]=bot_response
	botBlock["sender"]='bot'

	messages.append(botBlock)
	MESSAGES=history
	print(MESSAGES)

	with open('../msgHistory/patients/'+patient_name+'.json', 'w') as fs:
		json.dump(history,fs)

	current=open('../msgHistory/currentChat.txt','w')
	current.write(patient_name)
	current.close()

	return render_template('patient.html', MESSAGES=MESSAGES)

@app.route('/patientprocess', methods=['POST'])
def patientprocess():
	current = open('../msgHistory/currentChat.txt','r')
	patient_name=current.read()
	with open('../msgHistory/patients/'+patient_name+'.json') as f:
			history=json.load(f)
	PATIENTINDEX=history["PATIENTINDEX"]
	user_response = request.form['user_input'].lower()
	userBlock={}
	userBlock["message"]=user_response
	userBlock["sender"]='user'
	if PATIENTINDEX<7 and PATIENTINDEX>-1:
		data = patientIntake(PATIENTINDEX)[0]
		history[data]=user_response
		PATIENTINDEX=PATIENTINDEX+1
		if PATIENTINDEX==7:
			bot_response=patientIntake(PATIENTINDEX)[0]
		else:
			bot_response=patientIntake(PATIENTINDEX)[1]
	else:
		bot_response=response(user_response)

	messages=history["messages"]
	history["PATIENTINDEX"]=PATIENTINDEX

	botBlock={}
	botBlock["message"]=bot_response
	botBlock["sender"]='bot'

	messages.append(userBlock)
	messages.append(botBlock)
	MESSAGES=history
	print(MESSAGES)

	with open('../msgHistory/patients/'+patient_name+'.json', 'w') as fs:
		json.dump(history,fs)


	return render_template('patient.html', MESSAGES=MESSAGES)


@app.route('/process', methods = ['POST'])
def process():

	with open('../msgHistory/msg0.json') as f:
		history = json.load(f)
	messages=history["messages"]
	user_input=request.form['user_input'].lower()
	userBlock = {}
	userBlock["message"]=user_input
	userBlock["sender"]='user'
	if('can you look for' in user_input):
		results = search('fever')
		id_list = results['IdList']
		papers = fetch_details(id_list)
		bot_response=''
		for i, paper in enumerate(papers['PubmedArticle']): 
			bot_response=bot_response + "%d) %s" % (i+1, paper['MedlineCitation']['Article']['ArticleTitle'])+'\n'
			try:
				bot_response=bot_response + paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]+'\n'
			except Exception as e:
				bot_response = bot_response+"No Abstract\n"
	else:
		bot_response = response(user_input)
		print(bot_response)

	botBlock = {}
	botBlock["message"]=bot_response
	botBlock["sender"]='bot'

	messages.append(userBlock)
	messages.append(botBlock)
	MESSAGES=history
	print(MESSAGES)

	with open('../msgHistory/msg0.json', 'w') as fs:
		json.dump(history,fs)

	return render_template('doctor.html',user_input=user_input, bot_response=bot_response, MESSAGES=MESSAGES)

if __name__=='__main__':
	app.run(debug=True,port=5002)






















