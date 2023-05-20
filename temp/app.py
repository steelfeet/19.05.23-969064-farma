import sys, os, pathlib, time, random

import pdfkit

# ---------------------------------- VARIABLES
DATA_FILE = "Атлас 2021.pdf"

path_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'

# ---------------------------------- / VARIABLES

config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# директория файла
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "Исходники")

data_path = os.path.join(DATA_DIR, DATA_FILE)
html_path = os.path.join(BASE_DIR, "temp.html")

# Read the PDF file
pdf_file = open(data_path, 'rb')
# Convert the PDF to HTML
html_file = pdfkit.from_file(pdf_file, html_path, configuration=config)
# Close the PDF file
pdf_file.close()