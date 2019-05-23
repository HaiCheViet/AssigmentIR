import json
import math
from collections import Counter
from collections import defaultdict
from tqdm import tqdm


# LOAD DOCS AND VOCABULARY

with open('docs_list.txt') as docs_file:
    doc_lines = docs_file.read().splitlines()
    docs_list = [line.split() for line in doc_lines]

with open('vocab.words.txt') as vocab_file:
    lines = vocab_file.read().splitlines()
    vocab_list = [[line.split()[0], int(line.split()[1])] for line in lines]
# print(vocab_list)
# exit()

dictionary = {}
for i in tqdm(range(len(vocab_list))):
    term_dict = {}
    tf_overall = vocab_list[i][1]
    posting_dict = {}
    for j in range(len(docs_list)):
        tf = docs_list[j].count(vocab_list[i][0])
        if tf != 0:
            posting_dict[j] = {'tf': tf}
    num_of_docs = len(posting_dict)
    idf = math.log10(len(docs_list)/float(num_of_docs))
    dictionary[vocab_list[i][0]] = {'tf_overall': vocab_list[i][1], 'number_of_docs': num_of_docs,
                                'idf' : idf, 'posting_list' : posting_dict}
def norm(doc_index, doc):
    return math.sqrt(sum((dictionary[term]['idf']*dictionary[term]['posting_list'][doc_index]['tf'])**2 for term in set(doc)))

norm_list = [norm(doc_index, docs_list[doc_index]) for doc_index in range(len(docs_list))]            
for term in dictionary.keys():
    for doc_index in dictionary[term]['posting_list'].keys():
        dictionary[term]['posting_list'][doc_index]['w'] = dictionary[term]['idf']*dictionary[term]['posting_list'][doc_index]['tf']/norm_list[doc_index]

with open('dictionary.json', 'w') as dict_file:
    json.dump(dictionary, dict_file, indent=4)