# Identificação de discurso de ódio e linguagem ofensiva em tuítes usando SVM e CNN
Este projeto aplica dois classificadores, SVM e CNN, a um banco de tuítes rotulados para identificação de discurso de ódio e linguagem ofensiva.

A base de dados pode ser encontrada em https://github.com/t-davidson/hate-speech-and-offensive-language
O artigo que propõe a arquitetura de CNN testada pode ser encontrado em https://github.com/pinkeshbadjatiya/twitter-hatespeech
O modelo de embeddings (GloVe Twitter) pode ser encontrado em https://nlp.stanford.edu/projects/glove/

Requerimentos: Python 3.7, Keras, Numpy, Matplotlib, NLTK, CSV

Informe o diretório da base de dados no seu computador em pre_process.py e o diretório do embedding em emb_badja.py.
Execute main.py
