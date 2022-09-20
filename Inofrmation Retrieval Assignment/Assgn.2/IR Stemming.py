import os
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def tokenize(doc):
    doc_rp = remove_punctuation(doc).split()
    vocab = make_vocab(doc_rp)
    print(f'Vocabulary: {vocab}')
    print(f'Words in Vocab: {len(vocab)}\n')
    return vocab


def remove_punctuation(s1):
    for i in s1:
        if i in string.punctuation:
            s1 = s1.replace(i, "")
    return s1


def make_vocab(doc):
    vocab = []
    for y in doc:
        if y not in vocab:
            vocab.append(y)
    return vocab


def remove_stopwords(doc):
    doc_rsw = []
    sw = []
    stop_words = set(stopwords.words('english'))
    for w in doc:
        if w not in stop_words:
            doc_rsw.append(w)
        if w in stop_words:
            sw.append(w)
    print(f'***Removed Stop Words***\n{doc_rsw}')
    print(f'Words: {len(doc_rsw)}')
    print(f'\nRemoved {len(doc) - len(doc_rsw)} stop words, i.e.: {sw}\n')
    return doc_rsw


def stemming(doc):
    ps = PorterStemmer()
    print(f'***Stemming***')
    for w in doc:
        print(w, " : ", ps.stem(w))


os.getcwd()
f = open("news.txt", 'r')
read_file = f.read().lower()
word_token = tokenize(read_file)
x = remove_stopwords(word_token)
stemming(sorted(x))
