import os
import glob

DATA_DIR = "data/aclImdb/"
folders = ["test/neg/", "test/pos/", "test/unsup/",
           "train/neg/", "train/pos/", "train/neg/"]


def load_all_docs():
    docs = []
    for folder in folders:
        for g in glob.glob(DATA_DIR + folder + "*"):
            print(g)
            with open(g, 'r') as f:
                docs.append(f.readlines())
    return docs


print(load_all_docs())
