import csv
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


def pre_processamento(st='porter', base='david'):
    tt_texto = []  # Lista de tuites
    tt_label = []  # Lista de rotulos

    nltk.download('punkt')
    nltk.download('stopwords')

    snowball_stemmer = SnowballStemmer('english')
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    # Lê o arquivo tt_label.csv e armazena o conteudo numa lista.
    if base == 'waseem':
        with open('tt_label.csv') as csvfile:
            tweets = csv.reader(csvfile,  delimiter=',')
            for line in tweets:
                if line[1] == 'none':
                    tt_texto.append(line[0])
                    tt_label.append(0)
                elif line[1] == 'sexism':
                    tt_texto.append(line[0])
                    tt_label.append(1)
                elif line[1] == 'racism':
                    pass  # Retira os tuites rotulados como 'racismo'
    elif base == 'david':
        hatesp, offlan, none = 0, 0, 0
        with open('davidson_data.csv') as csvfile:
            tweets = csv.reader(csvfile,  delimiter=',')
            for line in tweets:
                tt_texto.append(line[6])
                if line[5] == '0':
                    tt_label.append(int(line[5]))
                    hatesp += 1
                elif line[5] == '1':
                    tt_label.append(int(line[5]))
                    offlan += 1
                elif line[5] == '2':
                    tt_label.append(int(line[5]))
                    none += 1
        print('A base possui ', hatesp, ' DO, ', offlan, ' LO e ', none, ' NE.')

    tt_label = np.array(tt_label)
    tt_pre = []  # Lista que ira conter os os tu[ites pre-processados
    maxlen = 0  # Contador de termos no tuíte pre-processado

    for tt in tt_texto:
        tt = tt.split(' ')  # Separa o texto pelo espaco

        # Retira os RT, os @, as # e os links
        remove_words = []
        for word in tt:
            word.lower()
            if word.startswith('@') | word.startswith('@', 1) | word.startswith('rt') | word.startswith('#') | \
                    word.startswith('&') | word.startswith('http'):
                remove_words.append(word)
        for rem_word in remove_words:
            tt.remove(rem_word)

        tt = ' '.join(tt)  # Junta o texto de novo numa string
        tt = word_tokenize(tt)  # Tokenizacao
        tt = [w for w in tt if w not in stop_words]  # Retira os stopwords
        tt = [w for w in tt if w.isalpha()]  # Retira pontuacao

        # Stemming sb = snow ball, pt = porter, ss = sem stemmer
        if st == 'snowball':
            for j, word in enumerate(tt):
                tt[j] = snowball_stemmer.stem(word)
        elif st == 'porter':
            for j, word in enumerate(tt):
                tt[j] = porter_stemmer.stem(word)
        else:
            pass
        if len(tt) > maxlen:
            maxlen = len(tt)
        tt = ' '.join(tt)  # Junta o texto de novo numa string
        tt_pre.append(tt)  # Adiciona o tuite pre-processado a lista

    return tt_pre, tt_label, maxlen
