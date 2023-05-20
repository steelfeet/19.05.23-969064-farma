import sys, os, pathlib, time, random
from tqdm import tqdm
import PyPDF2

# ---------------------------------- VARIABLES
DATA_FILE = "Атлас 2021.pdf"



# ---------------------------------- / VARIABLES


# директория файла
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "Исходники")


pdf_path = os.path.join(DATA_DIR, DATA_FILE)
file = open(pdf_path,'rb')
pdfReader = PyPDF2.PdfReader(file)
# printing number of pages in pdf file
print(f"Total number of pages in {DATA_FILE}", len(pdfReader.pages))

full_text = ""
for i in tqdm(range(0, len(pdfReader.pages))):
    # creating a page object
    pageObj = pdfReader.pages[i]
    # extracting text from page
    text = pageObj.extract_text()
    text = text.replace(" -\n", "")
    text = text.replace("\n", " ")
    full_text += text

txt_path = os.path.join(BASE_DIR, "temp.txt")
txt_file = open(txt_path,'w', encoding="utf-8")
txt_file.write(full_text)
txt_file.close()

