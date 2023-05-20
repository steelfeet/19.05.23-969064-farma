import sys, os, pathlib, time, random
from tqdm import tqdm


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

# ---------------------------------- VARIABLES
DATA_FILE = "Атлас 2021.pdf"



# ---------------------------------- / VARIABLES


# директория файла
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "Исходники")
TRAIN_DIR = os.path.join(BASE_DIR, "data")


with open(os.path.join(BASE_DIR, "farma_full.txt"), encoding="utf-8") as f:
    farma_dict = [line.rstrip() for line in f]

txt_path = os.path.join(BASE_DIR, "temp.txt")
txt_file = open(txt_path,'r', encoding="utf-8")
full_text = txt_file.readlines()[0]

# подчищаем от переносов строк и пр. артефактов
full_text = full_text.replace(" -", "")
full_text = full_text.replace(" - ", "")
full_text = full_text.replace("- ", "")

txt_file.close()

print("---------> Проверяем наличие растений в литературе <------------")
for farma_item in farma_dict:
    farma_item = farma_item.replace("\n", " ")
    is_exist = False

    if farma_item.lower() in full_text.lower():
        is_exist = True

    if not(is_exist):
        print(f"{farma_item} не найдено")
  

sentences_dict = full_text.split(".")
print(f"len sentences_dict {len(sentences_dict)}")


print("---------> Парсим текст <------------")
TRAIN_FILE = "красная_книга"
import joblib
clf = joblib.load(pathlib.Path(TRAIN_DIR, f"all_clf.pkl"))
vectorizer = joblib.load(pathlib.Path(TRAIN_DIR, f"all_vectorizer.pkl"))
tfidf_transformer = joblib.load(pathlib.Path(TRAIN_DIR, f"all_tfidf_transformer.pkl"))



def add_in_X(add_in_X):
    pass
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc 

emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
segmenter = Segmenter()


html = "<html><body><table>"
for sentence in tqdm(sentences_dict):
    for farma_item in farma_dict:
        farma_item = farma_item.replace("\n", " ").lower()
        if farma_item in sentence.lower():
            current_farma_item = farma_item

    tokens = preprocess_text(sentence)
    
    X_new = vectorizer.transform([tokens])
    X_new_tfidf = tfidf_transformer.transform(X_new)
    answer = clf.predict(X_new_tfidf)[0]

    # Красная книга 
    if int(answer) == 1:
        # Локации
        doc = Doc(sentence)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        locs = [org.text for org in doc.spans if org.type =='LOC']
        if len(locs) > 0:
            add_in_X(sentence)


