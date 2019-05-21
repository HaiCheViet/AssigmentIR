import re
from glob import glob
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle


glove = np.load("glove.npz")["embeddings"]

def get_dense(word_to_idx, word):
    try:
        idx = word_to_idx[word]
    except KeyError:
        return np.zeros((300,))
    return glove[idx]

def read_list(file="outfile"):
    with open (file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

# def


if __name__ == "__main__":

    with open("../vocab.words.txt", "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    save_vector = []

    list_data = glob("../clean_data/*")
    for n in list_data:
        idx = os.path.basename(n).split(".")[0]
        with open(n, "r") as f:
            doc = f.read()
            sentence_vector = sum([get_dense(word_to_idx, i) for i in doc.split()])
            if np.isnan(sentence_vector).any():
                print(sentence_vector)
            save_vector.append((idx, sentence_vector))

    with open('outfile', 'wb') as fp:
        pickle.dump(save_vector, fp)