import numpy as np

# Exibe os tuítes que foram corretamente ou erroneamente classificados em classes de interesse


def mostra_erros(y_test, y_pred, test_list, tt_pre):

    np.array(y_pred)
    for i, label in enumerate(y_test):

        if (label[0] == 1) & (np.argmax(y_pred[i, :]) == 0):
            print('ACERTO DO: ', tt_pre[test_list[i]])
        if (label[0] == 1) & (np.argmax(y_pred[i, :]) == 1):
            print('ERRO DO->LO: ', tt_pre[test_list[i]])
        if (label[1] == 1) & (np.argmax(y_pred[i, :]) == 1):
            pass
            # print('ACERTO LO: ', tt_pre[test_list[i]])
        if (label[1] == 1) & (np.argmax(y_pred[i, :]) == 0):
            print('ERRO LO->DO: ', tt_pre[test_list[i]])
