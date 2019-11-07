import glob
from load_files import get_word_counts, files_to_wordlists, file_to_wordlist
import os
import math
from nltk.stem.porter import PorterStemmer
import svmlight
from collections import Counter
import time


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

    def train(self, pos_word_lists, neg_word_lists):
        if self.stemming:
            porter_stemmer = PorterStemmer()
            pos_word_lists = [[porter_stemmer.stem(
                x) for x in l] for l in pos_word_lists]

            neg_word_lists = [[porter_stemmer.stem(
                x) for x in l] for l in neg_word_lists]

        if self.bigrams:
            if self.unigrams:
                neg_word_lists = [zip(docwords, docwords[1:]) + docwords
                                  for docwords in neg_word_lists]
                pos_word_lists = [zip(docwords, docwords[1:]) + docwords
                                  for docwords in pos_word_lists]
            else:
                neg_word_lists = [zip(docwords, docwords[1:])
                                  for docwords in neg_word_lists]
                pos_word_lists = [zip(docwords, docwords[1:])
                                  for docwords in pos_word_lists]

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

    def evaluate(self, test_files, targets):
        corrects = []
        data = []
        for i, wordlist in enumerate(files_to_wordlists(test_files)):
            if self.stemming:
                porter_stemmer = PorterStemmer()
                wordlist = [porter_stemmer.stem(w) for w in wordlist]
            if self.bigrams:
                if self.unigrams:
                    wordlist = zip(wordlist, wordlist[1:]) + wordlist
                else:
                    wordlist = zip(wordlist, wordlist[1:])
            if self.presence:
                wordlist = set(wordlist)
            c = Counter(wordlist)
            l = []
            for word, v in c.iteritems():
                if self.word_ids.get(word, 0) == 0:
                    self.word_ids[word] = self.curr_id
                    self.curr_id += 1

                l.append((self.word_ids.get(word), v))
            l.sort(key=lambda x: x[0])
            data.append((1 if targets[i] == "POS" else -1, l))
        results = svmlight.classify(self.model, data)

        for i, r in enumerate(results):
            if (r < 0 and targets[i] == "NEG") or (r > 0 and targets[i] == "POS"):
                corrects.append(True)
            else:
                corrects.append(False)
        return (corrects, float(sum(corrects))/len(corrects))


if __name__ == "__main__":
    pos = glob.glob("./POS/*")
    neg = glob.glob("./NEG/*")

    svm = SVM(presence=True)
    svm.train(pos_word_lists=files_to_wordlists(pos)[0:800],
              neg_word_lists=files_to_wordlists(neg)[0:800])
    print(svm.evaluate(pos[801:] + neg[801:], ["POS"]
                       * len(pos[801:]) + ["NEG"] * len(neg[801:]))[1])
