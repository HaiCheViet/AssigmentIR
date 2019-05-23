from clean_data import normalize
import os
from tqdm import tqdm
import re
import json
from collections import defaultdict
import operator

def input_from_keyboard():
    query = input()
    return query
def query_from_file(file_name):
    with open(file_name) as f:
        queries_list = f.read().splitlines()
        # print(queries_list[0])
        # exit()
    queries_list = {re.split('\t', query)[0] : re.split('\t', query)[-1] for query in queries_list if query != ''}
    for key, value in queries_list.items():
        queries_list[key] = normalize(value).split()
        result = compute_relevant(queries_list[key])
        with open(os.path.join(os.getcwd(), 'result', key+'.txt'), 'w') as f:
            for doc in result:
                f.write(key + ' ' + doc_name_list[int(doc)][:-4] + '\n')
            f.close()


def compute_relevant(query):
    relevant_docs = defaultdict(float)
    for term in query:
        if term in dictionary:
            tf = query.count(term)
            idf = dictionary[term]['idf']
            w = tf*idf
            for doc_index, doc in dictionary[term]['posting_list'].items():
                relevant_docs[doc_index] = relevant_docs[doc_index] + doc['w']*w
    sorted_relevant_docs = sorted(relevant_docs.items(), key=operator.itemgetter(1), reverse=True)
    ranked_docs = [sorted_relevant_docs[i][0] for i in range(len(sorted_relevant_docs))]
    return ranked_docs
            
doc_name_list = os.listdir(os.path.join(os.getcwd(), 'raw_data'))
with open('dictionary.json') as f:
    dictionary = json.load(f)
# query = input_from_keyboard()
query_from_file('query.txt')
# print(query)