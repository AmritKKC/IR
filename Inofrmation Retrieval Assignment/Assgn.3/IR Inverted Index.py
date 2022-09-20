import os
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def tokenize(doc):
    doc_rp = remove_punctuation(doc).split()
    return make_vocab(doc_rp)


def remove_punctuation(s1):
    for i in s1:
        if i in string.punctuation:
            s1 = s1.replace(i, " ")
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
    return sorted(doc_rsw)


path = str(os.getcwd()) + '\Docs'
os.chdir(path)
vocab = []
file_name = []
dict_ii = {}
for file in os.listdir():
    if file.endswith(".txt"):
        file_name.append(file)
        f = open(file, 'r')
        read_file = f.read().lower()
        vocab = remove_stopwords(tokenize(read_file))
        d = {}
        for word in vocab:
            d.update({word: (file, read_file.count(word))})
        for word in vocab:
            if word in dict_ii.keys():
                dict_ii.update({word: [dict_ii.get(word), d.get(word)]})
            else:
                dict_ii.update({word: d.get(word)})
print("\n***Inverted Index ***\n")
dict_ii = {key: val for key, val in sorted(dict_ii.items(), key=lambda ele: ele[0])}
for key in dict_ii.keys():
    print(key, " -> ", dict_ii.get(key))
