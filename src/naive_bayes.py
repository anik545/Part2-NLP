import glob
from load_files import get_word_counts, files_to_wordlists, file_to_wordlist
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

        self.negative_word_counts = get_word_counts(
            neg_word_lists, presence=self.presence)
        self.positive_word_counts = get_word_counts(
            pos_word_lists, presence=self.presence)

        if self.cutoff:
            self.negative_word_counts = Counter(
                {k: cutoff if v >= cutoff else v for k, v in self.negative_word_counts.iteritems()})
            self.positive_word_counts = Counter(
                {k: cutoff if v >= cutoff else v for k, v in self.positive_word_counts.iteritems()})

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

    def nb_classify_file(self, filename):
        posSum = 0
        negSum = 0
        words = file_to_wordlist(filename)
        if self.stemming:
            porter_stemmer = PorterStemmer()
            words = [porter_stemmer.stem(w) for w in words]

        if self.bigrams:
            words = zip(words, words[1:])

        if self.presence:
            words = set(words)

        for word in words:
            if word in self.word_log_probs:
                posSum += self.word_log_probs[word]["POS"]
                negSum += self.word_log_probs[word]["NEG"]
        posSum += self.class_probs["POS"]
        negSum += self.class_probs["NEG"]
        return "POS" if (posSum > negSum) else "NEG"

    def evaluate(self, test_files, targets):
        corrects = []
        for i, file in enumerate(test_files):
            if self.nb_classify_file(file) == targets[i]:
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
