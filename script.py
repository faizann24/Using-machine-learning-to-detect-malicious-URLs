
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.linear_model import LogisticRegression


def getTokens(input):
	tokensBySlash = str(input.encode('utf-8')).split('/')
	allTokens = []
	for i in tokensBySlash:
		tokens = str(i).split('-')
		tokensByDot = []
		for j in range(0,len(tokens)):
			tempTokens = str(tokens[j]).split('.')
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokens + tokensByDot
	allTokens = list(set(allTokens))
	if 'com' in allTokens:
		allTokens.remove('com')
	return allTokens

allurls = 'C:\\Users\\Faizan Ahmad\\Desktop\\Url Classification Project\\Data to Use\\allurls.txt'
allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)
allurlsdata = pd.DataFrame(allurlscsv)

allurlsdata = np.array(allurlsdata)

badurls = []

alltokens = []

s = ''
for i in allurlsdata:
	if i[1] == 'bad':
		badurls.append(i[0])
	tokens = getTokens(str(i[0]))
	for j in tokens:
		alltokens.append(j)
		

print alltokens.count('virus')

#s = str(badurls)
#wc = WordCloud(max_words=100).generate(s)
#plt.imshow(wc)
#plt.show()

random.shuffle(allurlsdata)

y = [d[1] for d in allurlsdata]
corpus = [d[0] for d in allurlsdata]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
X = vectorizer.fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print len(y_test)
print len(y_train)

lgs = LogisticRegression()
lgs.fit(X_train, y_train)
print(lgs.score(X_test, y_test))

X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
X_predict = vectorizer.transform(X_predict)
y_Predict = lgs.predict(X_predict)
print y_Predict

dataset = np.ndarray(shape=(2,3),dtype=np.float32)


#xx,yy = X_test.shape
#test = np.reshape(test, (2,yy))

#print test

#rint lgs.predict(X_test)







'''
vector = CountVectorizer(tokenizer=getTokens)
data = vector.fit_transform(string).toarray()
print data'''

