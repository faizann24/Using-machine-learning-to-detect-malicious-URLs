
import math
import random
import sys
import os

import pandas as pd
import numpy as np


from collections import Counter
from flask import Flask, request, Response, render_template
from functools import wraps
from subprocess import call

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)


def entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def getTokens(input):
	allTokens = list(set([str(i).split('.') for i in [str(i).split('-') for i in str(input.encode('utf-8')).split('/')]]))
	if 'com' in allTokens:
		allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
	return allTokens

def TL():
	allurlsdata = np.array(pd.DataFrame(pd.read_csv("./data/data.csv", ",", error_bad_lines=False)))
	random.shuffle(allurlsdata)	#shuffling

	y = [d[1] for d in allurlsdata]	#all labels 
	corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
	vectorizer = TfidfVectorizer(tokenizer=getTokens)	#get a vector for each url but use our customized tokenizer
	X = vectorizer.fit_transform(corpus)	#get the X vector

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

	lgs = LogisticRegression()	#using logistic regression
	lgs.fit(X_train, y_train)
	print(lgs.score(X_test, y_test))	#print the score. It comes out to be 98%
	return vectorizer, lgs

@app.route('/<path:path>')
def show_index(path):
	X_predict = vectorizer.transform([str(path)])
	y_Predict = lgs.predict(X_predict)
	return '''
		You asked for %s
		AI output: %s 
		Entropy: %s 
	''' % (path, str(y_Predict), str(entropy(path)))	

port = os.getenv('VCAP_APP_PORT', 5000)
if __name__ == "__main__":
	vectorizer, lgs  = TL()
	app.run(host='0.0.0.0',port=int(port), debug=True)

#vectorizer, lgs  = TL()
#checking some random URLs. The results come out to be expected. The first two are okay and the last four are malicious/phishing/bad

#X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
#X_predict = vectorizer.transform(X_predict)
#y_Predict = lgs.predict(X_predict)
#print(y_Predict)	#printing predicted values
