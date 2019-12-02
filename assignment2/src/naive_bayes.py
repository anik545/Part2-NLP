import glob
from load_files import get_word_counts, files_to_wordlists
import os
import math
from nltk.stem.porter import PorterStemmer
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class NaiveBayes(object):

    def __init__(self, presence=False, bigrams=False, cutoff=None, stemming=False, unigrams=True):
        self.presence = presence
        self.bigrams = bigrams
        self.cutoff = cutoff
        self.stemming = stemming
        self.unigrams = unigrams

    def train(self, docs):
        pos_word_lists = [doc[2] for doc in docs if doc[0] == "POS"]
        neg_word_lists = [doc[2] for doc in docs if doc[0] == "NEG"]


        self.negative_word_counts = get_word_counts(
            neg_word_lists, presence=self.presence)
        self.positive_word_counts = get_word_counts(
            pos_word_lists, presence=self.presence)

        self.vocabulary = set(
            self.negative_word_counts.keys() + self.positive_word_counts.keys())

        self.word_log_probs = self.get_log_probs()

        len_pos = len(pos_word_lists)
        len_neg = len(neg_word_lists)
        total = len_neg + len_pos

        self.class_probs = {
            "POS": math.log(float(len_pos)/total),
            "NEG": math.log(float(len_neg)/total)
        }

    def get_log_probs(self):
        total_pos = sum(self.positive_word_counts.values())
        total_neg = sum(self.negative_word_counts.values())
        total_unique_words = len(self.vocabulary)
        probs = {}

        for word in self.vocabulary:
            word_prob = {
                "POS": math.log((float(self.positive_word_counts[word]+1))/(total_pos+total_unique_words+1)),
                "NEG": math.log((float(self.negative_word_counts[word]+1))/(total_neg+total_unique_words+1))
            }
            probs[word] = word_prob
        return probs

    def nb_classify_file(self, words):
        posSum = 0
        negSum = 0

        for word in words:
            if word in self.word_log_probs:
                posSum += self.word_log_probs[word]["POS"]
                negSum += self.word_log_probs[word]["NEG"]
        posSum += self.class_probs["POS"]
        negSum += self.class_probs["NEG"]
        return "POS" if (posSum > negSum) else "NEG"

    def evaluate(self, docs):
        corrects = []
        for i, file in enumerate(docs):
            if self.nb_classify_file(file[2]) == file[0]:
                corrects.append(True)
            else:
                corrects.append(False)
        return (corrects, float(sum(corrects))/len(corrects))


if __name__ == "__main__":
    pos = glob.glob("./POS/*")
    neg = glob.glob("./NEG/*")

    nb = NaiveBayes(presence=True, bigrams=True, unigrams=False)
    nb.train(pos_word_lists=files_to_wordlists(pos)[0:900],
             neg_word_lists=files_to_wordlists(neg)[0:900])
    print(nb.evaluate(pos[901:] + neg[901:], ["POS"]
                      * len(pos[901:]) + ["NEG"] * len(neg[901:]))[1])
