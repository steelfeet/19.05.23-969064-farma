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



X_text_1 = []
# Красная Книга - 1
delin_path = pathlib.Path(TRAIN_DIR, f"красная_книга.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]
for block in sentences:
    X_text_1.append(block)


# Период посева - 2
X_text_2 = []
delin_path = pathlib.Path(TRAIN_DIR, f"posev.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]
for block in sentences:
    X_text_2.append(block)


# Период сбора - 3
X_text_3 = []
delin_path = pathlib.Path(TRAIN_DIR, f"period_sbora.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]
for block in sentences:
    X_text_3.append(block)


# Содержание БАВ, хим состав - 4
X_text_4 = []
delin_path = pathlib.Path(TRAIN_DIR, f"sostav.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]
for block in sentences:
    X_text_4.append(block)

# Ежегодная потребность лекарственного сырья ,тонны - 6
X_text_6 = []
delin_path = pathlib.Path(TRAIN_DIR, f"potreb.txt") 
with open(delin_path, "r", encoding='utf-8') as f:
	sentences = [line.rstrip() for line in f]
for block in sentences:
    X_text_6.append(block)

def add_in_X(sentence, num):
    # if num == 1:
    pass




from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc 

emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
segmenter = Segmenter()

monthes = ["январь", "февраль", "март", "май", "апрель", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
html = "<html><body><table><tr align=center><td>Название</td><td>Красная книга</td><td>Регион Красной книги</td><td>Период посева, мес</td><td>Период сбора урожая, мес</td><td>Содержание БАВ, хим состав</td><td>Препараты</td><td>Ежегодная потребность лекарственного сырья ,тонны</td></tr>"
clear_html = "<html><body><table><tr align=center><td>Название</td><td>Красная книга</td><td>Регион Красной книги</td><td>Период посева, мес</td><td>Период сбора урожая, мес</td><td>Содержание БАВ, хим состав</td><td>Препараты</td><td>Ежегодная потребность лекарственного сырья ,тонны</td></tr>"

html_1 = "" # КК
html_1r = "" # Регион
dict_1r = [] # Регион
html_2 = "" # Посев
html_3 = "" # Период
html_4 = "" # Состав
html_5 = "" # Препарат
html_6 = "" # Потребность
html_1s = "" # КК
html_2s = "" # Посев
html_3s = "" # Период
html_4s = "" # Состав
html_5s = "" # Препарат
html_6s = "" # Потребность
current_farma_item = ""

bgcolor = "gray"
for sentence in tqdm(sentences_dict):
    if len(sentence) > 30:
        for farma_item in farma_dict:
            farma_item = farma_item.replace("\n", " ").lower()
            if farma_item in sentence.lower():
                if farma_item != current_farma_item:
                    for loc in dict_1r:
                        html_1r += f"{loc}, "
                    
                    if bgcolor == "white":
                        bgcolor = "lime"
                    else:
                        bgcolor = "white"
                    html += f"<tr valign=top bgcolor={bgcolor}><td>{current_farma_item}</td><td>{html_1}<hr>{html_1s}</td><td>{html_1r}</td><td>{html_2}<hr>{html_2s}</td><td>{html_3}<hr>{html_3s}</td><td>{html_4}<hr>{html_4s}</td><td>{html_5}<hr>{html_5s}</td><td>{html_6}<hr>{html_6s}</td></tr>"
                    clear_html += f"<tr valign=top bgcolor={bgcolor}><td>{current_farma_item}</td><td>{html_1}</td><td>{html_1r}</td><td>{html_2}</td><td>{html_3}</td><td>{html_4s}</td><td>{html_5}</td><td>{html_6}</td></tr>"
                    html_1 = ""
                    html_1r = ""
                    dict_1r = [] # Регион
                    html_2 = ""
                    html_3 = ""
                    html_4 = ""
                    html_5 = ""
                    html_6 = ""
                    html_1s = "" 
                    html_2s = ""
                    html_3s = ""
                    html_4s = ""
                    html_5s = ""
                    html_6s = ""
                    html += f"</tr>"
                    current_farma_item = farma_item

        tokens = preprocess_text(sentence)
        
        X_new = vectorizer.transform([tokens])
        X_new_tfidf = tfidf_transformer.transform(X_new)
        answer = clf.predict(X_new_tfidf)[0]

        # Красная книга 
        if int(answer) == 1:
            # Локации
            is_good = False
            if ("красная" in tokens) and ("книга" in tokens):
                html_1 = f"да"
                add_in_X(sentence, 1)
                doc = Doc(sentence)
                doc.segment(segmenter)
                doc.tag_ner(ner_tagger)
                locs = [org.text for org in doc.spans if org.type =='LOC']
                if len(locs) > 0:
                    html_1s += f"{sentence}<br>----<br>"
                    for loc in locs:
                        if loc not in dict_1r:
                            dict_1r.append(loc)
                

        # Период посева
        if int(answer) == 2:
            html_2s += f"{sentence}<br>----<br>"
            is_good = False
            for month in monthes:
                if month in tokens:
                    html_2 += f"{month}, "
                    is_good = True
            if is_good:
                add_in_X(sentence, 2)


        # Период сбора
        if int(answer) == 3:
            html_3s += f"{sentence}<br>----<br>"
            is_good = False
            for month in monthes:
                if month in tokens:
                    html_3 += f"{month}, "
                    is_good = True
            if is_good:
                add_in_X(sentence, 3)

        # состав
        if int(answer) == 4:
            html_4s += f"{sentence}<br>----<br>"

        # Препараты
        if int(answer) == 5:
            html_5s += f"{sentence}<br>----<br>"
            doc = Doc(sentence)
            doc.segment(segmenter)
            doc.tag_ner(ner_tagger)
            locs = [org.text for org in doc.spans]
            if len(locs) > 0:
                for loc in locs:
                    html_5 += f"{loc}, "


        # Потребность
        if int(answer) == 6:
            html_6s += f"{sentence}<br>----<br>"
            is_good = False
            if " т " in sentence:
                add_in_X(sentence, 6)
                
                



with open(os.path.join(BASE_DIR, "temp.html"), "w", encoding="utf-8") as f:
    f.write(html)
f.close()

with open(os.path.join(BASE_DIR, "clear.html"), "w", encoding="utf-8") as f:
    f.write(clear_html)
f.close()
