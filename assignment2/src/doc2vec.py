from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from load_files import files_to_wordlists, load_all_docs, load_all_pang_docs
from cross_validate import validation_set
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    '''Callback to save model after each epoch and show training parameters '''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class Doc2VecSVM(object):

    def __init__(self, doc2vec_args={}):
        self.doc2vec_model = None
        self.svm_model = None
        self.doc2vec_args = doc2vec_args

    def train_doc_vec(self, docs):
        documents = tqdm([TaggedDocument(doc, [i]) for i, doc in enumerate(docs)], desc="Loading tagged docs into model")
        model = Doc2Vec(documents, vector_size=100, window=2,
                        min_count=1, workers=4, seed=0, callbacks=[EpochLogger()], **self.doc2vec_args)
        model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)
        self.doc2vec_model = model

        with open("/mnt/c/Users/Anik/Files/Work/units/NLP/assignment2/models/model_"+str(self.doc2vec_args), 'w') as f:
            model.save(f)

    def set_model(self, model_fname):
        self.doc2vec_model = Doc2Vec.load(model_fname)

    # TODO: make this match up to old svm code
    def train_svm(self, docs):
        # docs is a list of (pos/neg, filename, wordlist)
        svmlight_lines = []
        for doc in docs:
            fv = list(enumerate(self.doc2vec_model.infer_vector(doc[2])))
            svmlight_line = (1 if doc[0] == "POS" else -1, fv)
            svmlight_lines.append(svmlight_line)

        self.svm_model = learn(svmlight_lines)

    def train(self, svm_train_docs, doc2vec_train_docs):
        self.train_doc_vec(doc2vec_train_docs)
        print("*** DOC2VEC TRAINED ***")
        self.train_svm(svm_train_docs)
        print("*** SVM TRAINED ***")

    # test_docs: list of (pos/neg, filename, wordlist)
    def evaluate(self, test_docs, targets):
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


if __name__ == "__main__":
    pang_docs = load_all_pang_docs()
    all_docs = load_all_docs()
    validation, the_rest = validation_set(pang_docs)

    model = Doc2VecSVM()
    model.train(the_rest, all_docs)
    c, p = model.evaluate(validation)

    print(p)

    # next set of args
    # model = Doc2VecSVM()
    # model.train(the_rest, all_docs)
    # model.evaluate(validation)

    # ... etc.
