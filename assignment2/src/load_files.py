import os
import glob
from tqdm.notebook import tqdm
from collections import Counter

DATA_DIR = "/mnt/c/Users/Anik/Files/Work/units/NLP/data/aclImdb/"

folders = ["test/neg/", "test/pos/", "test/unsup/",
           "train/neg/", "train/pos/", "train/unsup/"]


def _load_all_docs():
    docs = []
    for folder in folders:
        print(folder)
        for g in glob.glob(DATA_DIR + folder + "*"):
            with open(g, 'r') as fp:
                words = [word for line in fp for word in line.split()]
                docs.append(words)
    print("*** LOADED DOCUMENTS FOR DOC2VEC ***")
    return docs


def load_all_docs():
    with open("/mnt/c/Users/Anik/Files/Work/units/NLP/assignment2/all_docs.txt", "r") as f:
        docs = []
        for line in tqdm(f.readlines(), desc="Loading files"):
            docs.append(line.split(' '))

    print("*** LOADED DOCUMENTS FOR DOC2VEC ***")
    return docs


PANG_DATA_DIR = "/mnt/c/Users/Anik/Files/Work/units/NLP/data/part1/"

pang_folders = ['NEG', 'POS']


def load_all_pang_docs(presence=False):
    docs = []
    for folder in pang_folders:
        for fp in tqdm(glob.glob(PANG_DATA_DIR + folder + '/*'), desc="Loading files from "+folder):
            with open(fp, 'r') as f:
                words = [word for line in f for word in line.split()]
                if presence:
                    words = set(words)
            docs.append((folder, fp, words))
    return docs


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

if __name__ == "__main__":
    load_all_pang_docs()
