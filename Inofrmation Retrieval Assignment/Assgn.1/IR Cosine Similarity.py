import os
import string
import math
import numpy as np
from numpy.linalg import norm


def remove_punctuation(s1):
    for i in s1:
        if i in string.punctuation:
            s1 = s1.replace(i, "")
    return s1


def make_vocab(lists):
    vocab = []
    for x in lists:
        for y in x:
            if y not in vocab:
                vocab.append(y)
    return vocab


def calc_freq(v, f):
    dic = {}
    print("\n------ Frequency in Documents ------")
    for word in v:
        freq = []
        for doc in f:
            freq.append(doc.count(word))
        dic.update({word: freq})
        print(word, ":", freq)
    return dic


def calc_tf(freq):
    print("\n----- Term Frequency -----")
    for key in freq:
        ls = []
        for value in freq.get(key):
            if value == 0:
                ls.append(value)
            if value > 0:
                ls.append(float(format(1 + math.log2(value), "0.3f")))
        freq.update({key: ls})
        print(key, ":", ls)
    return freq


def calc_idf(n, tfif):
    idf = {}
    print("\n------ Inverse Document Frequency ------")
    for k in tfif:
        count = 0
        for value in tfif.get(k):
            if value > 0:
                count = count + 1
        result = float(format(math.log2(n / count), "0.3f"))
        idf.update({k: result})
        print(k, ":", result)
    return idf


def calc_weight(tf, idf):
    print("\n------ Weight ------")
    for key in tf:
        wt = []
        for value in tf.get(key):
            result = float(format(value * idf.get(key), "0.3f"))
            wt.append(result)
        tf.update({key: wt})
        print(key, ":", wt)
    print("\n")
    return tf


def cos_sim(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


os.getcwd()
new_file = []
file_name = []
w_value = []
for file in os.listdir():
    if file.endswith(".txt"):
        file_name.append(file)
        f = open(file, 'r')
        read_file = f.read().lower()
        new_file.append(remove_punctuation(read_file).split())
print("Files: ", file_name)
n = len(new_file)
doc_no = int(0)
for i in new_file:
    print("Document ", doc_no + 1, ": ", i)
    doc_no += 1
vocab = make_vocab(new_file)
print("\nVocabulary: ", vocab)
fij = calc_freq(vocab, new_file)
tfij = calc_tf(fij)
idf = calc_idf(n, tfij)
w = calc_weight(tfij, idf)
for key in w:
    w_value.append(w.get(key))
z = np.transpose(w_value)
for i in range(len(file_name)):
    for j in range(len(file_name)):
        print(f'Cosine Similarity ({i+1},{j+1}) : {cos_sim(z[i], z[j])}')
