import pickle
import re
from collections import defaultdict
from glob import glob

import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from clean_data import normalize

glove = np.load("glove.npz")
glove = glove["embeddings"]


def get_dense(word_to_idx, word):
    try:
        idx = word_to_idx[word]
    except KeyError:
        print(word)
        return np.zeros((300,))
    return glove[idx]


def read_list(file="outfile"):
    with open(file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def init_normalize_query(path="../DEV-TEST/query.txt"):
    query = []

    with open(path, "r") as f:
        for i in f:
            item = re.split(r"\t+", i)
            if len(item) != 2:
                continue
            idx = item[0]
            sentence_predict = normalize(item[1])
            query.append((idx, sentence_predict))
    return query


def init_dev_eval(path="../DEV-TEST/RES/*"):
    dev_eval_list = glob(path)
    dev_eval = defaultdict(list)

    for file_path in dev_eval_list:
        with open(file_path, "r") as f:
            for i in f:
                item = re.split(r"\t+", i)[0].split()
                if len(item) != 2:
                    continue

                idx = item[0]
                match_file = item[1]
                dev_eval[idx].append(match_file)
    return dev_eval


def calculate_score(arr_1, arr_2):
    sm = len(set(arr_1) - set(arr_2))
    score = (1 - sm / len(arr_2)) * 100
    return score


def calculate_vector(word_to_idx, sentence_list):
    sentence_list = set(sentence_list)
    if len(sentence_list) == 0:
        return 0
    return sum([get_dense(word_to_idx, i) for i in sentence_list])/len(sentence_list)


if __name__ == "__main__":
    query = init_normalize_query()
    dev_eval = init_dev_eval()
    vector_trained = read_list()
    path_output = "../result_embedding/"
    result_score = []

    with open("vocab.words.txt", "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    for q in tqdm(query):
        idx_q = q[0]
        sentence_pred_vec = calculate_vector(word_to_idx, sentence_list=q[1].split())

        y_real = dev_eval[idx_q]

        temp_order_predict = {}

        for v in vector_trained:
            idx_v = v[0]
            doc = v[1]
            # print(doc[:len(q[1].split())])
            sentence_vector = calculate_vector(word_to_idx, doc)

            cos_sim = dot(sentence_vector, sentence_pred_vec) / (norm(sentence_vector) * norm(sentence_pred_vec))
            if isinstance(cos_sim, np.float64):
                temp_order_predict[idx_v] = cos_sim
            else:

                temp_order_predict[idx_v] = 0

        temp_order_predict = list(sorted(temp_order_predict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        # if idx_q == "1":
        #     print(temp_order_predict)
        y_pred = [i[0] for i in temp_order_predict]

        with open(path_output + f"{idx_q}.txt", "w") as f:
            for i in y_pred:
                f.write(f"{idx_q} {i}\n")
