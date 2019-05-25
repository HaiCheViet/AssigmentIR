

from clean_data import normalize
from export_result import calculate_vector, read_list
import numpy as np
from numpy import dot
from numpy.linalg import norm

glove = np.load("glove.npz")
glove = glove["embeddings"]

if __name__ == "__main__":
    input_q = str(input("Write your query here: "))
    vector_trained = read_list()
    y_pred = {}

    with open("vocab.words.txt", "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    input_q = normalize(input_q).split()
    sentence_pred_vec = calculate_vector(word_to_idx, input_q)

    for v in vector_trained:
        idx_v = v[0]
        doc = v[1][:len(input_q)]
        # if idx_v == "102":
        #     print(doc)
        sentence_vector = calculate_vector(word_to_idx, doc)

        cos_sim = dot(sentence_vector, sentence_pred_vec) / (norm(sentence_vector) * norm(sentence_pred_vec))
        if isinstance(cos_sim, np.float64):
            y_pred[idx_v] = cos_sim
        else:

            y_pred[idx_v] = 0

    y_pred = list(sorted(y_pred.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))[:20]
    for i in y_pred:
        print(f"{i[0]}\t{i[1]}")


