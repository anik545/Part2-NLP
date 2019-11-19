from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from load_files import files_to_wordlists, load_all_docs, load_all_pang_docs
from cross_validate import validation_set
from tqdm.notebook import tqdm
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


MODEL_DIR_PATH = "/mnt/c/Users/Anik/Files/Work/units/NLP/assignment2/models/"


class Doc2VecSVM(object):

    def __init__(self, doc2vec_args={}, doc2vec_model=None, svm_model=None):
        self.doc2vec_model = Doc2Vec.load(MODEL_DIR_PATH + doc2vec_model) if doc2vec_model else None
        self.svm_model = read_model(svm_model) if svm_model else None
        self.doc2vec_args = doc2vec_args

    def train_doc_vec(self, docs):
        documents = tqdm([TaggedDocument(doc, [i]) for i, doc in enumerate(
            docs)], desc="Loading tagged docs into model")
        model = Doc2Vec(documents, callbacks=[EpochLogger()], **self.doc2vec_args)
        model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)
        self.doc2vec_model = model

        with open("/mnt/c/Users/Anik/Files/Work/units/NLP/assignment2/models/model_"+str(self.doc2vec_args), 'w') as f:
            model.save(f)
        print("*** DOC2VEC TRAINED ***")

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

    def train(self, svm_train_docs, doc2vec_train_docs):
        self.train_doc_vec(doc2vec_train_docs)
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

        return (corrects, float(sum(corrects))/len(corrects))


def run_with_args(args, doc2vec_train, pang_svm_train, validation):
    defaults = {'vector_size':100, 'window':2, 'min_count':1, 'workers':4, 'seed':0}
    defaults.update(args)
    print(defaults)
    model = Doc2VecSVM(doc2vec_args=defaults)
    model.train(pang_svm_train, doc2vec_train)
    c, p = model.evaluate(validation)
    print(args, p)
    return (args,p)


if __name__ == "__main__":

    pang_docs = load_all_pang_docs()
    all_docs = load_all_docs()
    validation, the_rest = validation_set(pang_docs)
    # TODO maybe try using presence in loading pang_docs - does it even effect anything, since we're not using count vectors here?
    # Actually, probably shouldn't use presence here
    run_with_args({'dm':0,'epochs':10, 'vector_size':120, 'min_count':2, 'window':7}, all_docs, the_rest, validation)
    run_with_args({'dm':0,'epochs':10, 'vector_size':120, 'min_count':2, 'window':7, 'dbow_words': 1}, all_docs, the_rest, validation)