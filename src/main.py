import itertools
from cross_validation import cross_validate, sign_and_cv
from sign_test import sign_test
from naive_bayes import NaiveBayes
from svm import SVM


optionlist = [
    {"unigrams": True, "presence": False},
    {"unigrams": True, "presence": True},
    {"unigrams": True, "presence": True, "stemming": True},
    {"bigrams": True, "unigrams": False, "presence": True},
    {"bigrams": True, "unigrams": True, "presence": True}
]

for options in optionlist:
    print(options)
    acc, _ = cross_validate(NaiveBayes(**options),
                            ["./POS", "./NEG"], folds=10)
    print("Naive Bayes", " Accuracy: ", acc)
    acc, _ = cross_validate(SVM(**options), ["./POS", "./NEG"], folds=10)
    print("SVM", " Accuracy: ", acc)
    print()

comparisons = [
    (NaiveBayes(**{"unigrams": True, "presence": False}),
     SVM(**{"unigrams": True, "presence": False})),
    (SVM(**{"presence": False}),
     SVM(**{"presence": True})),
    (NaiveBayes(**{"presence": False}),
     NaiveBayes(**{"presence": True})),
    (NaiveBayes(**{"presence": True}),
     SVM(**{"presence": True})),
    (NaiveBayes(**{"unigrams": True, "bigrams": True}),
     SVM(**{"unigrams": True, "bigrams": True})),
    (NaiveBayes(**{"unigrams": True, "presence": True}),
     SVM(**{"bigrams": True, "presence": True}))
]

for modelA, modelB in comparisons:

    p, accA, accB = sign_and_cv(
        modelA, modelB,  ["./POS", "./NEG"], folds=10)
    print(p)
    print
