from cross_validation import cross_validate, sign_and_cv
from sign_test import sign_test
from naive_bayes import NaiveBayes
from svm import SVM


acc, _ = cross_validate(NaiveBayes(), ["./POS", "./NEG"])
print("Naive Bayes, frequency, unigrams", acc)

acc, _ = cross_validate(NaiveBayes(presence=True), ["./POS", "./NEG"])
print("Naive Bayes, presence, unigrams", acc)

acc, _ = cross_validate(NaiveBayes(
    presence=True, bigrams=True), ["./POS", "./NEG"])
print("Naive Bayes, presence, bigrams", acc)


acc, _ = cross_validate(SVM(), ["./POS", "./NEG"])
print("SVM, frequency, unigrams", acc)

p, accA, accB = sign_and_cv(NaiveBayes(), SVM(), ["./POS", "./NEG"])
