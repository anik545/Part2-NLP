import numpy as np

# return list of list of docs, each list is a split

def flatten(l):
    return [x for b in l for x in b]

def split_stratified(docs, n=10):
    pos_docs = [doc for doc in docs if doc[0] == "POS"]
    neg_docs = [doc for doc in docs if doc[0] == "NEG"]
    splits = [pos_docs[x::n] + neg_docs[x::n] for x in range(n)]
    return splits


def validation_set(docs, percent=10):
    splits = split_stratified(docs, n=100/percent)
    validation = splits[0]
    the_rest = flatten(splits[1:])

    return (validation, the_rest)

if __name__ == "__main__":
    s = validation_set([("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),
    ("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"])])
    print(s)