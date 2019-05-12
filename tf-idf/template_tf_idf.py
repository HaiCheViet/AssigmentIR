from collections import defaultdict
import math

def create_dic(word_ls,doc):

    return dic
dic = {
    "tag": {
        "STL": 2,
        "Ts" :,
        "idf": ,
        "DS_posting" : {
            0: {
                "tf": 2
                # "w": (tf*idf)/ norm[i-1]
            },
            1: {
                "tf": 3
            }
        }
    }
}
def norm(d1):
    return math.sqrt(sum([dic[i]["idf"] * dic[i]["DS_posting"][d1]["tf"])**2 for i in d1])

list_norm = [norm(i) for i in doc]

for i in dic:
    for doc_id in i["DS_posting"]:
        doc_id["w"] = i["idf"] * doc_id["tf"] / list_norm[doc_id]



dic_temp = defaultdict(int)
for i in q:
    w_chi_q = dic[i]["idf"] * Counter(i, q)
    for doc_id in dic[i]["DS_posting"]:
        w_chi_d = dic[i]["DS_posting"][doc_id]
        # doc_id id doc id
        w = w_chi_q * w_chi_d
        dic_temp[doc_id]+=w

for i in sorted(dic_temp.values()):
    print(i)
