import gensim
import numpy as np

DATA_DIR = "/mnt/c/Users/Anik/Files/Work/units/NLP/data"


def load_all_docs():
    docs = []
    for folder in folders:
        for g in glob.glob(DATA_DIR + folder + "*"):
            with open(f, 'r') as fp:
                words = [word for line in fp for word in line.split()]
                docs.append(words)

    return docs


PANG_DATA_DIR = "/mnt/c/Users/Anik/Files/Work/units/NLP/data/part1"

pang_folders = ['NEG', 'POS']


def load_all_pang_docs():
    docs = []
    for folder in pang_folders:
        for fp in glob.glob(PANG_DATA_DIR + folder + '/*'):
            with open(fp, 'r') as f:
                words = [word for line in f for word in line.split()]
            docs.append((folder, fp, words))
    return docs


class Doc2VecSVM(object):

    def __init__(self, doc2vec_args={}):
        self.doc2vec_model = None
        self.svm_model = None
        self.doc2vec_args = args

    def train_doc_vec(self, docs):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        model = Doc2Vec(documents, vector_size=100, window=2,
                        min_count=1, workers=1, seed=0, **self.doc2vec_args)
        model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)
        self.doc2vec_model = model

        with open("../models/model_+"str(args), 'w') as f:
            model.save(f)

    # TODO: make this match up to old svm code
    def train_svm(self, docs):
        # docs is a list of lists of (pos/neg, filename, wordlist)
        svmlight_lines = []
        for doc in docs:
            fv = list(enumerate(self.doc2vec_model.infer_vector(doc[2])))
            svmlight_line = (1 if doc[0] == "POS" else -1, fv)
            svmlight_lines.append(svmlight_line)

        self.svm_model = learn(svmlight_lines)

    def train(self, svm_train_docs, doc2vec_train_docs=load_all_docs()):
        train_doc_vec(doc2vec_train_docs)
        train_svm(svm_train_docs)

    def evaluate(self, test_files, targets):
        features = []
        for doc in docs:
            fv = list(enumerate(self.doc2vec_model.infer_vector(doc[2])))
            svmlight_line = (1 if doc[0] == "POS" else -1, fv)
            features.append(svmlight_line)

        predictions = classify(self.svm_model, features)
        predictions = ["POS" if p > 0 else "NEG" for p in predictions]

        corrects = [True if predictions[i] == targets[i]
                    else False for i in range(len(predictions))]

        return (corrects, float(sum(corrects))/len(corrects))
