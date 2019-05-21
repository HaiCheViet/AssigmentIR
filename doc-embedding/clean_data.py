import spacy
import os
import glob
from tqdm import tqdm
import re
nlp = spacy.load("en")


def remove_special_character(text):
    data = re.sub("[^a-z0-9@.\-']", " ", text)
    return data

def normalize(text):
    text = remove_special_character(text)
    doc = nlp(text)
    result = []
    for token in doc:
        # Remove punct and stop word
        if not token.is_stop and not token.is_punct:
            if token.lemma_ != " ":
                result.append(token.lemma_)

    return " ".join(result)

if __name__ == "__main__":
    list_data = glob.glob("../raw_data/*")
    for i in tqdm(list_data):
        with open(i, "r") as f:
            data = normalize(f.read())
            name_txt = os.path.basename(i)
            with open(f"../clean_data/{name_txt}", "w") as out:
                out.write(data)