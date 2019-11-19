import glob
from load_files import files_to_wordlists, load_all_docs, load_all_pang_docs
import os
import math
from nltk.stem.porter import PorterStemmer
import svmlight
from collections import Counter
import time
from cross_validate import validation_set


def flatten(l):
    return [x for b in l for x in b]


class SVM(object):

    def __init__(self, presence=False, bigrams=False, cutoff=None, stemming=False, unigrams=True):
        self.presence = presence
        self.bigrams = bigrams
        self.cutoff = cutoff
        self.stemming = stemming
        self.curr_id = 1
        self.unigrams = unigrams

    def train(self, docs):
        pos_word_lists = [doc[2] for doc in docs if doc[0] == "POS"]
        neg_word_lists = [doc[2] for doc in docs if doc[0] == "NEG"]
        vocabulary = set(flatten(pos_word_lists) + flatten(neg_word_lists))
        self.word_ids = {}
        a = []
        for k in vocabulary:
            self.word_ids[k] = self.curr_id
            self.curr_id += 1
        features = []
        for wordlist in pos_word_lists:

            if self.presence:
                wordlist = set(wordlist)

            c = Counter(wordlist)
            featureVec = [(self.word_ids.get(word), v)
                          for word, v in c.iteritems()]
            featureVec.sort(key=lambda x: x[0])
            features.append((1, featureVec))

        for wordlist in neg_word_lists:
            if self.presence:
                wordlist = set(wordlist)
            c = Counter(wordlist)
            featureVec = [(self.word_ids.get(word), v)
                          for word, v in c.iteritems()]
            featureVec.sort(key=lambda x: x[0])
            features.append((-1, featureVec))
        self.model = svmlight.learn(features)

    def evaluate(self, docs):
        corrects = []
        data = []
        for classification, fname, wordlist in docs:
            c = Counter(wordlist)
            l = []
            for word, v in c.iteritems():
                if self.word_ids.get(word, 0) == 0:
                    self.word_ids[word] = self.curr_id
                    self.curr_id += 1

                l.append((self.word_ids.get(word), v))
            l.sort(key=lambda x: x[0])
            data.append((1 if classification == "POS" else -1, l))
        results = svmlight.classify(self.model, data)

        for i, r in enumerate(results):
            if (r < 0 and docs[i][0] == "NEG") or (r > 0 and docs[i][0] == "POS"):
                corrects.append(True)
            else:
                corrects.append(False)
        return (corrects, float(sum(corrects))/len(corrects))


if __name__ == "__main__":
    # TODO make sure to reload all pang docs here to use presence
    pang_docs = load_all_pang_docs(presence=True)

    validation, train_data = validation_set(pang_docs)

    svm = SVM()
    svm.train(train_data)
    e,p = svm.evaluate(validation)
    print(p)
