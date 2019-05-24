import pickle
import re
from collections import defaultdict
from glob import glob

import numpy as np
from export_vector import get_dense
from numpy import dot
from numpy.linalg import norm

from clean_data import normalize


def read_list(file="outfile"):
    with open(file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def init_query(path="../DEV-TEST/query.txt"):
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
    dev_eval_list = glob("../DEV-TEST/RES/*")
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

def calculate_query(sentence):
    """
    Code transfer query in here
    :param sentence:
    :return:
    """
    result = None

    return result


if __name__ == "__main__":
    query = init_query()
    dev_eval = init_dev_eval()
    vector_trained = read_list()
    result_score = []


    with open("../vocab.words.txt", "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    for q in query:
        idx_q = q[0]
        sentence_pred_vec = calculate_query(q[1].split()) # sentence for query in here
        y_real = dev_eval[idx_q]

        temp_order_predict = {} # Score for each doc per query in here

        # Only get top doc similarity score with length == result doc in here
        y_pred = [i[0] for i in temp_order_predict[:len(y_real)]]

        # y_pred have to be same length with y_real
        result_score.append(calculate_score(y_real, y_pred))
    print(result_score)
    score = sum(result_score) / len(result_score)
    print(score)
