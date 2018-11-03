# coding=utf-8
from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# Carrega arquivo na memoria
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Transforma o documento em tokens limpos
def clean_doc(doc, vocab):
    tokens = doc.split()
    # Remove uma pontuaçao de cada token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # Filta tokens que nao estao no vocabulario
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# Carrega todos os documentos em um diretorio
def process_docs(directory, vocab, is_trian):
    documents = list()
    # Percorre todos os documentos dentro de um diretorio
    for filename in listdir(directory):
        # Pula qualquer comentario no nosso dataset
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue

        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc, vocab)
        documents.append(tokens)
    return documents


# Carrega um "embutido" tipo como um diretorio (E uma camada de incorporaçao que mapeia indices das palavras)
def load_embedding(filename):
    # Carrega o "embutido" na momoria e pula a primeira linha
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    # Cria um mapa de palavras para um vetor
    embedding = dict()
    for line in lines:
        parts = line.split()
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


# Peguei da internet, ainda estou estudado o que faz
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


# Um segunda maneira de obter a acuracia. Ainda nao foi testado
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# Carrega o vocaculario
vocab_filename = 'vocabulario.txt'  # Vocabulario feito por Edivaldo
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# Carrega todas as avaliaçoes de treinamento
# positive_docs = process_docs('txt_sentoken/pos', vocab, True)
positive_docs = process_docs('Dataset/Banco de Dados Positivos', vocab, True)
negative_docs = process_docs('Dataset/Banco de Dados Negativos', vocab, True)
train_docs = negative_docs + positive_docs

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Definiçao de rotulos de treinamento
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# Carrega toda a revisao de testes que foram feitos
positive_docs = process_docs('Dataset/Banco de Dados Positivos', vocab, False)
negative_docs = process_docs('Dataset/Banco de Dados Negativos', vocab, False)
test_docs = negative_docs + positive_docs

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# Definiçao de rotulos de treinamento
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

vocab_size = len(tokenizer.word_index) + 1

# Carregar Incorporaçao de um arquivo
raw_embedding = load_embedding('glove.6B.100d.txt')  #-  Realizar alguns teste
# obter vetores na ordem certa
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# Definiçao de um modelo
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# Compilaçao da Rede
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# Avaliaçao feita por meio da Acuracia!
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc * 100))
