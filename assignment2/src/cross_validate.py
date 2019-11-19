import numpy as np
from permutation_test import permutation_test
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


def cross_validate(system, docs):
    n=10
    splits = split_stratified(docs, n=n)
    corrects = []
    ps = []
    for x in range(n):
        train = flatten(splits[:x] + splits[(x+1):])
        test = splits[x]
        system.train(train)
        c, p = system.evaluate(test)
        corrects.append(c)
        ps.append(p)
    return np.mean(np.array(ps)), corrects

def permutation_test_and_cv(systemA, systemB, docs):
    pA, correctsA = cross_validate(systemA, docs)
    pB, correctsB = cross_validate(systemB, docs)

    print("Accuracy of A: ", pA)
    print("Accuracy of B: ", pB)

    p = permutation_test(correctsA, correctsB)
    # B beats A with p-value of p
    return p, pA, pB

if __name__ == "__main__":
    s = validation_set([("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),
    ("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"]),("POS","a",["a","b"])])
    print(s)