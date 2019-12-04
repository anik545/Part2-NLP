import os
import glob
from tqdm import tqdm
from collections import Counter
import nltk

DATA_DIR = "/home/ar899/Part2-NLP/data/aclImdb/"
PANG_DATA_DIR = "/home/ar899/Part2-NLP/data/part1/"
ALL_DOCS_FILE = "/home/ar899/Part2-NLP/assignment2/all_docs.txt"

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
    with open(ALL_DOCS_FILE, "r") as f:
        docs = []
        for line in tqdm(f.readlines(), desc="Loading files"):
            docs.append(nltk.word_tokenize(line.decode("utf-8").strip("\n")))
    print("*** LOADED DOCUMENTS FOR DOC2VEC ***")
    return docs


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
