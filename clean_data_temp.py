import spacy
import os
import glob
from tqdm import tqdm
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


snowball_stemmer = SnowballStemmer(language = 'english')
wordnet_lemmatizer = WordNetLemmatizer()

stop_file = open('stopwords_en.txt')
stop_list = list(stop_file.read().splitlines())
stop_file.close()

def remove_special_character(text):
    data = re.sub(r"[^a-z]+", " ", text)
    return data

def normalize(text):
    text = text.lower()
    text = remove_special_character(text).split()
    result = []
    for token in text:
        if token not in stop_list:
            token = wordnet_lemmatizer.lemmatize(token)
            token = snowball_stemmer.stem(token)
            if token != ' ':
                result.append(token)

    return " ".join(result)


list_data = glob.glob("raw_data/*")
for i in tqdm(list_data):
    with open(i, "r") as f:
        data = normalize(f.read())
        name_txt = os.path.basename(i)
        with open(f"clean_data/{name_txt}", "w") as out:
            out.write(data)