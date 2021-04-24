import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score


def executa_tfidf(tt_pre, param_svm):

    # Variáveis de controle
    ngram_min = param_svm[0]
    ngram_max = param_svm[1]
    mindf = param_svm[2]

    #  Cria instancia da classe CountVectorizer que faz a tokenizacao e cria a matriz documento-termo
    contador = CountVectorizer(tt_pre, stop_words='english', ngram_range=(ngram_min, ngram_max), min_df=mindf)
    matriz_doc_termo = contador.fit_transform(tt_pre)
    vocabulario = contador.get_feature_names()
    # print(contador.get_feature_names())  # Exibe o vocabulario do corpus

    #  Cria instancia da classe TfidfTransformer que faz calculo tfidf sobre a matriz doc-termo
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf_matriz_doc_termo = tfidf.fit_transform(matriz_doc_termo)

    # Exibe os termos mais frequentes em ordem
    # somamatriz = numpy.sum(matriz_doc_termo, axis=0)
    # indmax = numpy.argmax(somamatriz)
    # print(contador.get_feature_names()[indmax])
    # words_freq = [(word, somamatriz[idx]) for word, idx in contador.vocabulary_.items()]
    # words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    # print(words_freq)

    return vocabulario, tfidf_matriz_doc_termo


def executa_svm(tt_label, tfidf_matriz_doc_termo, tam_teste=0.3):

    # Faz a divisao entre o conjunto de teste e o conjunto de treino
    train_list, test_list, x_train, x_test, y_train, y_test = train_test_split(range(len(tt_label)),
                                                                               tfidf_matriz_doc_termo,
                                                                               tt_label,
                                                                               test_size=tam_teste,
                                                                               random_state=111)

    # Cria o classificador, faz a validação cruzada e a predição
    classificador = svm.LinearSVC(class_weight='balanced', random_state=111)
    scores = cross_val_score(classificador, x_train, y_train, cv=5, scoring='f1_macro')
    classificador.fit(x_train, y_train)
    z1 = classificador.predict(x_test)

    # Exibe as metricas da classificação
    print(metrics.confusion_matrix(y_test, z1))
    # print("macro:", metrics.precision_recall_fscore_support(y_test, z1, average='macro'))
    print("micro:", metrics.precision_recall_fscore_support(y_test, z1, average='micro'))
    print(metrics.matthews_corrcoef(y_test, z1))

    # Exibe as matrizes de confusao
    metrics.plot_confusion_matrix(classificador, x_test, y_test,
                                  display_labels=['DO', 'LO', 'NE'],
                                  normalize='pred', cmap=plt.cm.Blues)

    metrics.plot_confusion_matrix(classificador, x_test, y_test,
                                  display_labels=['DO', 'LO', 'NE'],
                                  normalize='true', cmap=plt.cm.Purples)
    # plt.show()

    # Trecho do código dedicado a reduzir a dimensionalidade dos dados para visualização
    # pca = TruncatedSVD(n_components=3).fit(x_train)
    # pca_2d = pca.transform(x_train)
    # pca_2d_test = pca.transform(x_test)
    # svm_classifier_2d = svm.LinearSVC(random_state=111).fit(pca_2d, y_train)
    # z2 = svm_classifier_2d.predict(pca_2d_test)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for k in range(len(y_test)):
    #     if y_test[k] == 0:
    #         c1 = ax.scatter(pca_2d_test[k, 0], pca_2d_test[k, 1], pca_2d_test[k, 2], c='r', s=10, marker='o')
    #     elif y_test[k] == 1:
    #         # c2 = ax.scatter(pca_2d_test[k, 0], pca_2d_test[k, 1], pca_2d_test[k, 2], c='g', s=10, marker='+')
    #     else:
    #         c3 = ax.scatter(pca_2d_test[k, 0], pca_2d_test[k, 1], pca_2d_test[k, 2], c='b', s=10, marker='*')
    # pylab.legend([c1, c2, c3], ['DO', 'LO', 'NE'])
    # pylab.title('Visualização em 3D do conjunto de teste')
    # pylab.show()

    return train_list, test_list
