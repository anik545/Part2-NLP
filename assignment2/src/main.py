from src.load_files import load_all_pang_docs, load_all_docs
from src.doc2vec import Doc2VecSVM
from src.cross_validate import cross_validate, validation_set, permutation_test
from src.svm import SVM
from src.naive_bayes import NaiveBayes

pang_docs = load_all_pang_docs()
validation, the_rest = validation_set(pang_docs)

pang_docs_pres = load_all_pang_docs(presence=True)
validation_pres, the_rest_pres = validation_set(pang_docs_pres)

doc2vecsvm = Doc2VecSVM(doc2vec_model="model_best")
p, cs = cross_validate(model, the_rest)

svm_freq = SVM()
p1, svm_f = cross_validate(svm_freq, the_rest)


svm_pres = SVM()
p1, svm_p = cross_validate(svm_pres, the_rest_pres)

nb_freq = NaiveBayes()
p1, nb_f = cross_validate(nb_freq, the_rest)

nb_pres = NaiveBayes()
p1, nb_p = cross_validate(nb_pres, the_rest_pres)


pv_svmf = permutation_test(cs,svm_f) # 0.001
pv_svmp = permutation_test(cs,svm_p) # 0.44 !!
pv_nbf = permutation_test(cs,nb_f) # 0.001
pv_nbp = permutation_test(cs,nb_p) # 0.002

print(pv_svmf,pv_svmp,pv_nbf,pv_nbp)