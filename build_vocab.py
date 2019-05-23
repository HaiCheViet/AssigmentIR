"""Script to build words, chars and tags vocab"""

from collections import Counter
from pathlib import Path
from glob import glob

# TODO: modify this depending on your needs (1 will work doc_idust fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    list_data = glob("clean_data/*")
    print('Build vocab words (may take a while)')
    counter_words = Counter()
    docs_list = []
    with open('docs_list.txt', 'w') as docs_list_f:
        for n in list_data:
            with Path(n).open() as f:
                words = f.read()
                docs_list_f.write(words+'\n')
                words = words.strip().split()
                counter_words.update(words)
                

    filtering = {w:c for w, c in counter_words.items() if c >= MINCOUNT}
    sort = sorted(filtering.items(), key= lambda x:x[0])


    with Path('vocab.words.txt').open('w') as f:
        for w, c in sort:
            f.write('{} {}\n'.format(w,c))
    print('- done. Kept {} out of {}'.format(
        len(filtering), len(counter_words)))
