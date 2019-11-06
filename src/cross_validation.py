from naive_bayes import NaiveBayes
from os import listdir, path
import itertools
from load_files import files_to_wordlists
from svm import SVM
from sign_test import sign_test

# dirs = (posdir,negdir)


def cross_validate(model, dirs, folds=3):
    nfolds = [() for n in range(folds)]
    for n in range(folds):
        # (pos, neg)
        nfolds[n] = ((sorted(listdir(dirs[0]))[n::folds]),
                     sorted(listdir(dirs[1]))[n::folds])
    scores = []
    corrects_per_fold = []
    for i, fold in enumerate(nfolds):
        train_fold_set = {n for n in range(folds)} - {i}
        train_fold_pos = list(itertools.chain(*[nfolds[x][0]
                                                for x in train_fold_set]))
        train_fold_neg = list(itertools.chain(*[nfolds[x][1]
                                                for x in train_fold_set]))

        train_fold_pos = list(
            map(lambda p: str(path.join(dirs[0], p)), train_fold_pos))
        train_fold_neg = list(
            map(lambda p: str(path.join(dirs[1], p)), train_fold_neg))

        model.train(pos_word_lists=files_to_wordlists(train_fold_pos),
                    neg_word_lists=files_to_wordlists(train_fold_neg))

        targets = ["POS" for j in range(
            len(nfolds[i][0]))] + ["NEG" for j in range(len(nfolds[i][1]))]
        test_files = [path.join(dirs[0], f) for f in nfolds[i][0]] + \
            [path.join(dirs[1], f) for f in nfolds[i][1]]

        corrects, percent = model.evaluate(test_files, targets)
        corrects_per_fold.append(corrects)
        scores.append(percent)
        print("Testing on fold", i, "accuracy:", percent)
    print("Average accuracy: ", sum(scores)/len(scores))
    return (sum(scores)/len(scores)), corrects_per_fold


def sign_and_cv(modelA, modelB, dirs, folds=3):
    cvA = cross_validate(modelA, dirs)
    cvB = cross_validate(modelB, dirs)

    print("Accuracy of A: ", cvA[0])
    print("Accuracy of B: ", cvB[0])

    fold_corrects_A = [i for l in cvA[1] for i in l]
    fold_corrects_B = [i for l in cvB[1] for i in l]

    # p_values = []
    # for f1, f2 in zip(fold_corrects_A, fold_corrects_B):
    #     p_values.append(sign_test(f1, f2))
    # print("A beats B with p value of:", min(p_values))
    p = sign_test(fold_corrects_A, fold_corrects_B)
    return p, cvA[0], cvB[0]


# print(cross_validate(NaiveBayes(), ["./POS", "./NEG"])[0])
# print(cross_validate(SVM(), ["./POS", "./NEG"])[0])
sign_and_cv(NaiveBayes(), SVM(), ["./POS", "./NEG"])
