import scandir as os
import os
from collections import Counter


def file_to_wordlist(path, transform=lambda x: x):
    with open(path, 'r') as f:
        lines = f.readlines()
        data = list(map(lambda w: transform(w.replace('\n', '')), lines))
    return data


def files_to_wordlists(paths, transform=lambda x: x):
    data = []
    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            data.append(
                list(map(lambda w: transform(w.replace('\n', '')), lines)))
    data = list(filter(lambda x: x != '', data))
    return data

def get_word_counts(wordlists, presence=False):
    if presence:
        wordlists = [set(l) for l in wordlists]
    word_counts = Counter()
    for l in wordlists:
        word_counts.update(l)
    return word_counts
