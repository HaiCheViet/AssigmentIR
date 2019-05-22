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


if __name__ == "__main__":
    query = init_query()
    dev_eval = init_dev_eval()
    vector_trained = read_list()
    result_score = []


    with open("../vocab.words.txt", "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    for q in query:
        idx_q = q[0]
        sentence_pred_vec = sum([get_dense(word_to_idx, i) for i in set(q[1].split())])
        y_real = dev_eval[idx_q]

        temp_order_predict = {}

        for vector_doc in vector_trained:
            idx_v = vector_doc[0].split(".")[0]
            sentence_vector = vector_doc[1]

            cos_sim = dot(sentence_vector, sentence_pred_vec) / (norm(sentence_vector) * norm(sentence_pred_vec))
            if isinstance(cos_sim, np.float64):
                temp_order_predict[idx_v] = cos_sim
            else:
                temp_order_predict[idx_v] = 0

        temp_order_predict = list(sorted(temp_order_predict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        y_pred = [i[0] for i in temp_order_predict[:len(y_real)]]
        # if calculate_score(y_real, y_pred) == 0:
        #         #     print(temp_order_predict[:len(y_real)])
        #         #     print(y_real)
        #         #     print(idx_q)
        #         #     break
        result_score.append(calculate_score(y_real, y_pred))
    print(result_score)
    score = sum(result_score) / len(result_score)
    print(score)
