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
    for n in list_data:
        with Path(n).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))
