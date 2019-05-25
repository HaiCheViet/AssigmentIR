import os
import pickle
from glob import glob

import math
from textblob import TextBlob as tb
from tqdm import tqdm


def read_list(file="outfile"):
    with open(file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tf_idf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


if __name__ == "__main__":

    save_vector = []
    vector_trained = []

    list_data = glob("../clean_data/*")
    for n in list_data:
        idx = os.path.basename(n).split(".")[0]
        with open(n, "r") as f:
            doc = f.read()
            doc = tb(str(doc))
            vector_trained.append(doc)

    for n in tqdm(list_data):
        idx = os.path.basename(n).split(".")[0]
        with open(n, "r") as f:
            doc = f.read()
            blob = tb(doc)
            scores = {word: tf_idf(word, blob, vector_trained) for word in blob.words}
            sorted_words = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:30]
            doc = [i[0] for i in sorted_words]
            if not doc:
                print(idx)
                continue

            save_vector.append((idx, doc))

    with open('outfile', 'wb') as fp:
        pickle.dump(save_vector, fp)
