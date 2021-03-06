from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from load_files import files_to_wordlists, load_all_docs, load_all_pang_docs
from cross_validate import validation_set
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from svmlight import learn, classify, read_model


class EpochLogger(CallbackAny2Vec):
    '''Callback to save model after each epoch and show training parameters '''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


MODEL_DIR_PATH = "/home/ar899/Part2-NLP/assignment2/models/"


class Doc2VecSVM(object):

    def __init__(self, doc2vec_args={}, doc2vec_model=None, svm_model=None, doc2vec_train_docs=None):

        self.doc2vec_train_docs = doc2vec_train_docs
        self.doc2vec_model = Doc2Vec.load(
            MODEL_DIR_PATH + doc2vec_model) if doc2vec_model else None
        self.svm_model = read_model(svm_model) if svm_model else None
        self.doc2vec_args = doc2vec_args

    def train_doc_vec(self, docs):
        if docs is None:
            docs = load_all_docs()
            self.doc2vec_train_docs = docs

        documents = tqdm([TaggedDocument(doc, [i]) for i, doc in enumerate(docs)],
                         desc="Loading tagged docs into model")

        model = Doc2Vec(documents, callbacks=[
                        EpochLogger()], **self.doc2vec_args)

        self.doc2vec_model = model

        print("*** DOC2VEC TRAINED ***")

    def args_str(self, args):
        s = ""
        for a, v in args.items():
            s += str(a) + str(v) + "_"
        return s

    def set_model(self, model_name):
        self.doc2vec_model = Doc2Vec.load(MODEL_DIR_PATH + model_name)

    def train_svm(self, docs):
        # docs is a list of (pos/neg, filename, wordlist)
        svmlight_lines = []
        for doc in docs:
            fv = list(enumerate(self.doc2vec_model.infer_vector(doc[2]), 1))
            svmlight_line = (1 if doc[0] == "POS" else -1, fv)
            svmlight_lines.append(svmlight_line)

        self.svm_model = learn(svmlight_lines)
        print("*** SVM TRAINED ***")

    def train(self, svm_train_docs):
        # only train the doc2vec once with a certain set of args
        if self.doc2vec_model is None:
            self.train_doc_vec(self.doc2vec_train_docs)
        self.train_svm(svm_train_docs)

    # test_docs: list of (pos/neg, filename, wordlist)
    def evaluate(self, test_docs):
        features = []
        for doc in test_docs:
            fv = list(enumerate(self.doc2vec_model.infer_vector(doc[2]), 1))
            svmlight_line = (1 if doc[0] == "POS" else -1, fv)
            features.append(svmlight_line)
        predictions = classify(self.svm_model, features)
        predictions = ["POS" if p > 0 else "NEG" for p in predictions]
        corrects = [True if predictions[i] == test_docs[i][0]
                    else False for i in range(len(predictions))]
        acc = float(sum(corrects))/len(corrects)
        return (corrects, acc)


def run_with_args(args, pang_svm_train, validation, all_docs=None):
    defaults = {'vector_size': 100, 'window': 2, 'min_count': 1,
                'workers': 4, 'seed': 0, 'dbow_words': 1}
    defaults.update(args)
    print(defaults)
    model = Doc2VecSVM(doc2vec_args=defaults, doc2vec_train_docs=all_docs)
    model.train(pang_svm_train)
    c, p = model.evaluate(validation)
    print(args, p)
    return (args, p)


if __name__ == "__main__":

    pang_docs = load_all_pang_docs()
    validation, the_rest = validation_set(pang_docs)
    # TODO maybe try using presence in loading pang_docs - does it even effect anything, since we're not using count vectors here?
    # Actually, probably shouldn't use presence here
    # run_with_args({'dm':0,'epochs':10, 'vector_size':120, 'min_count':2, 'window':7}, the_rest, validation)
    run_with_args({'dm': 0, 'epochs': 5, 'vector_size': 120, 'min_count':   15,
                    'window': 7, 'dbow_words': 1, 'hs': 1}, the_rest, validation)
    # run_with_args({'dm': 0, 'epochs': 7, 'vector_size': 150, 'min_count': 18,
    #                'window': 7, 'dbow_words': 1, 'hs': 1}, the_rest, validation)
    # run_with_args({'dm': 0, 'epochs': 10, 'vector_size': 130, 'min_count': 14,
    #                'window': 6, 'dbow_words': 1, 'hs': 1}, the_rest, validation)
