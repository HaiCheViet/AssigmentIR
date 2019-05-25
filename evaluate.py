import os
from tqdm import tqdm

path = os.getcwd()
query_index_list = os.listdir(os.path.join(path, r'DEV-TEST/RES'))

def read_real_file(text):
    text = text.replace('\t',' ')
    lines = text.splitlines()
    real_doc_list = [int(line.split()[1]) for line in lines if line != '']
    real_doc_list.pop()
    return real_doc_list

def read_predict_file(text):
    lines = text.splitlines()
    predict_doc_list = [int(line.split()[1]) for line in lines]
    return predict_doc_list

AP_list = []
for file_name in tqdm(query_index_list):
    with open(os.path.join(path, r'DEV-TEST/RES', file_name)) as f1:
        real_doc_list = read_real_file(f1.read())
        f1.close()
    with open(os.path.join(path,'result_embedding',file_name)) as f2:
        predict_doc_list = read_predict_file(f2.read())
        f2.close()
    expected_relevant_doc_number = len(real_doc_list)
    non_interpolating_r = []
    non_interpolating_p = []
    counter = 1
    for i in range(len(predict_doc_list)):
        if predict_doc_list[i] in real_doc_list:
            non_interpolating_r.append(counter/expected_relevant_doc_number)
            non_interpolating_p.append(counter/(i+1))
            counter += 1
            if counter > expected_relevant_doc_number:
                break

    interpolating_r = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interpolating_p = []
    index = 0
    for i in range(len(interpolating_r)):
        while (index < len(non_interpolating_r) and non_interpolating_r[index] < interpolating_r[i]):
            index += 1
        if index >= len(non_interpolating_r):
            break
        interpolating_p.append(max(non_interpolating_p[index:]))
    AP = sum(interpolating_p)/11.0
    AP_list.append(AP)

print(AP_list)
MAP = sum(AP_list)/len(AP_list)
print(MAP)
        

    