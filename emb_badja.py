import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Softmax, \
    GlobalMaxPooling1D, Concatenate
from keras import Sequential
from confusao import matriz_conf
from erros import mostra_erros


def executa_emb(vocabulario, maxlen, tt_pre, tt_label, train_list, test_list, param_cnn):

    # Variáveis de controle
    embedding_dim = param_cnn[0]
    batch = param_cnn[1]
    epochs = param_cnn[2]
    n_filters = param_cnn[3]
    kernels = param_cnn[4]
    val_split = param_cnn[5]
    trainable = param_cnn[6]
    pesos = param_cnn[7]

    lossfn = 'categorical_crossentropy'
    class_names = ['DO', 'LO', 'NE']
    if pesos == 'sim':
        class_weight = {0: 5.78, 1: 0.43, 2: 1.98}
    else:
        class_weight = None

    # Lê o arquivo do GloVe embedding e cria um dicionario dos vocabulos e suas representações
    file_dict = {25: "/Volumes/Documentos/GloVe/glove/glove.twitter.27B.50d.txt",
                 50: "/Volumes/Documentos/GloVe/glove/glove.twitter.27B.50d.txt",
                 100: "/Volumes/Documentos/GloVe/glove/glove.twitter.27B.100d.txt",
                 200: "/Volumes/Documentos/GloVe/glove/glove.twitter.27B.200d.txt"}

    embeddings_dict = {}
    with open(file_dict[embedding_dim], 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    acertos = 0
    erros = 0
    faltando = []

    # Cria um dicionario do vocabulario do corpus e o índice de cada vocábulo
    # Prepara a embedding matrix associando cada vacábulo do cospus ao vetor do embedding
    # Conta quantos vocábulos do corpus estão presentes no dicionario do GloVe
    voc_index = dict(zip(vocabulario, range(len(vocabulario))))
    num_tokens = len(vocabulario) + 2
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in voc_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            acertos += 1
        else:
            faltando.append(word)
            erros += 1
    print("Converteu %d palavras (%d erros). Taxa de acertos = %.3f" % (acertos, erros, erros/(acertos+erros)))
    # print('Vocabulario: ', vocabulario)
    # print('Faltando: ', faltando)

    # Cria as matrizes de entrada e saída para treino e teste
    x_train = np.zeros((len(train_list), maxlen))
    x_test = np.zeros((len(test_list), maxlen))

    y_train = np.zeros((len(train_list),))
    y_test = np.zeros((len(test_list),))

    for i, index in enumerate(train_list):
        y_train[i] = tt_label[index]
        tt = tt_pre[index].split(' ')
        for j, word in enumerate(tt):
            x_train[i][j] = voc_index.get(word, 1)

    for i, index in enumerate(test_list):
        y_test[i] = tt_label[index]
        tt = tt_pre[index].split(' ')
        for j, word in enumerate(tt):
            x_test[i][j] = voc_index.get(word, 0)

    # Transforma os vetores de saída para o formato "one-hot encoding"
    # Formato exigido pela função de perda categorical crossentropy
    y_train = to_categorical(y_train, len(class_names))
    y_test = to_categorical(y_test, len(class_names))

    ###########################################################
    # Programação da rede neural proposta por Badjatiya e outros (2017)

    # Camada de embedding
    embedding_layer = Embedding(num_tokens,
                                embedding_dim,
                                input_length=maxlen,
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=trainable)

    # Camada de convolução
    graph_in = Input(shape=(maxlen, embedding_dim))
    convs = []
    for ksz in kernels:
        conv = Conv1D(filters=n_filters,
                      kernel_size=ksz,
                      padding='valid',
                      activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)
    out = Concatenate(axis=-1)(convs)
    graph = Model(graph_in, out)

    # Montagem das camadas
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.25))
    model.add(graph)
    model.add(Dropout(0.50))
    model.add(Activation('relu'))
    model.add(Dense(len(class_names)))
    model.add(Activation('softmax'))
    model.summary()

    # Treino
    model.compile(loss=lossfn,
                  # optimizer='adam',
                  optimizer="rmsprop",
                  metrics=[Precision(class_id=0), Recall(class_id=0),
                           Precision(class_id=1), Recall(class_id=1),
                           Precision(class_id=2), Recall(class_id=2),
                           Precision()])

    historico = model.fit(x_train, y_train,
                          batch_size=batch, epochs=epochs,
                          validation_split=val_split,
                          class_weight=class_weight)
    # print(historico.history.keys())

    # Teste
    print('###############################################')
    model.evaluate(x_test, y_test, batch_size=batch)
    y_predito = model.predict(x_test, batch_size=batch)

    # Exibe exemplos de tuítes classificados erroneamente
    # mostra_erros(y_test, y_predito, test_list, tt_pre)

    # Exibe gráfico da evolução do treinamento
    fig = plt.figure()
    ax = fig.gca()
    plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
    # plt.xticks([9, 19, 29, 39, 49], ['10', '20', '30', '40', '50'])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.grid()

    plt.plot(historico.history['val_precision'], 'b')
    plt.plot(historico.history['val_recall'], 'b--')

    plt.plot(historico.history['val_precision_1'], 'r')
    plt.plot(historico.history['val_recall_1'], 'r--')

    plt.plot(historico.history['val_precision_2'], 'g')
    plt.plot(historico.history['val_recall_2'], 'g--')
    # plt.plot(historico.history['loss'])
    # plt.plot(historico.history['val_loss'])

    plt.legend(['prec DO', ' rec DO',
                'prec LO', 'rec LO',
                'prec NE', 'rec NE',
                'perda treino', 'perda valid.'],
               loc='upper left')
    # plt.legend(['perda treino', 'perda valid.'])

    # plt.title('Evolução do treinamento')
    # plt.ylabel('Perdas')
    plt.ylabel('Precisão e recall')
    plt.ylim(0, 1)
    plt.xlabel('Época')
    plt.show()
