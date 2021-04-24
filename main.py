from pre_process import pre_processamento
from exec_svm import executa_tfidf
from exec_svm import executa_svm
from emb_badja import executa_emb

# Parâmetros SVM
ngram_min = 1
ngram_max = 2
mindf = 1
tam_teste = 0.3
param_svm = [ngram_min, ngram_max, mindf]

# Parâmetros CNN
embedding_dim = 200
batch = 128
epochs = 5
n_filters = 100
kernels = [1, 2, 3]
val_split = 0.2
trainable = True
pesos = 'nao'
param_cnn = [embedding_dim, batch, epochs, n_filters, kernels, val_split, trainable, pesos]

tt_pre, tt_label, maxlen = pre_processamento()
vocabulario, tfidf_matriz_doc_termo = executa_tfidf(tt_pre, param_svm)
train_list, test_list = executa_svm(tt_label, tfidf_matriz_doc_termo, tam_teste)
executa_emb(vocabulario, maxlen, tt_pre, tt_label, train_list, test_list, param_cnn)
