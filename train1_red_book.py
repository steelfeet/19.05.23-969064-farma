# -*- encoding: utf-8 -*-
# 
import os, json, time
from tqdm import tqdm

# добавляем текущую директорую в PATH
import pathlib
import sys
# на уровень выше расположения файла
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = "красная_книга"
Y_DATA = 1


from sklearn.feature_extraction.text import CountVectorizer
# download stopwords corpus, you need to run it once
import nltk
# nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
import pymorphy2
from string import punctuation







#Create lemmatizer and stopwords list
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

# для очистки всех ненужных нам знаков, которые вводит пользователь:
def preprocess_text(text):
	words = text.lower().split()
	
	# очистка от прилегающего к слову мусора (слово, "или так")
	clear_words = []
	for word in words:
		clear_word = ""
		for s in word:
			if not s in punctuation:
				clear_word = clear_word + s
		clear_words.append(clear_word)

	tokens = [morph.parse(token)[0].normal_form for token in clear_words if token not in russian_stopwords\
			and token != " " \
			and token.strip() not in punctuation]

	text = " ".join(tokens)
	
	return text


start_time = time.time()

# заполняем обучающий датасет
X_text = []
y = []
y_text = {}

# предложения по теме
delin_path = pathlib.Path(TRAIN_DIR, f"{TRAIN_FILE}.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]

print("len(sentences)", len(sentences))
for block in sentences:
	clear_phrase = preprocess_text(block)
	if len(clear_phrase) > 0:
		X_text.append(clear_phrase)
		y.append(1)

# предложения НЕ! по теме
sentences = []
delin_path = pathlib.Path(TRAIN_DIR, f"not_{TRAIN_FILE}.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	    content = f.read()
sentences = content.split('.')
print("len(BAD)", len(sentences))
for block in tqdm(sentences):
	clear_phrase = preprocess_text(block)
	if len(clear_phrase) > 0:
		X_text.append(clear_phrase)
		y.append(0)


vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_text)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=3)

"""
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1).fit(X_train_tfidf, y)


tokens = preprocess_text("На восточной границе своего ареала чага входит в Красную книгу Амурской области (2003).")
#tokens = preprocess_text("Содержание алкалоидов зависит от экологических факторов: специфики местообитания, температурного режима и др.")


X_new = vectorizer.transform([tokens])
X_new_tfidf = tfidf_transformer.transform(X_new)
answer = clf.predict(X_new_tfidf)[0]
print()
print(answer)
answer = clf.predict_proba(X_new_tfidf)[0]
print(answer)




import joblib
# save the classifier
joblib.dump(clf, pathlib.Path(TRAIN_DIR, f"{TRAIN_FILE}_clf.pkl"), compress=9)
joblib.dump(vectorizer, pathlib.Path(TRAIN_DIR, f"{TRAIN_FILE}__vectorizer.pkl"), compress=9)
joblib.dump(tfidf_transformer, pathlib.Path(TRAIN_DIR, f"{TRAIN_FILE}_tfidf_transformer.pkl"), compress=9)

print ("update time:", time.time()-start_time)

