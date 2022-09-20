import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import math
from numpy.linalg import norm

# Define Stopwords
stop_words = set(stopwords.words())


def pre_process(f):
    for i in f:
        token = word_tokenize(i.lower())
        token = [word.lower() for word in token if word not in stop_words]
        token = [word.lower() for word in token if word.isalpha()]
    return token


def make_vocab(lists):
    vocab = []
    for x in lists:
        for y in x:
            if y not in vocab:
                vocab.append(y)
    return vocab


def document_vector(v, p_file):
    dv = []
    for f in p_file:
        vec = [0] * len(v)
        for item in f:
            vec[v.index(item)] += 1
        dv.append(vec)
        del vec
    return dv


def cos_sim(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


os.getcwd()
file_name = []
processed_file = []
vocab = []
freq = []
for file in os.listdir():
    if file.endswith(".txt"):
        file_name.append(file)
        f = open(file, 'r')
        #   Pre-processing File
        processed_file.append(pre_process(f))


n = len(file_name)
print("Files: ", file_name)

#   Making Vocabulary
vocab = make_vocab(processed_file)
print("\nVocabulary: ", vocab)

# Creating Frequency
freq = document_vector(vocab, processed_file)
for t in range(n):
    print("\nFrequency for", file_name[t], "\n", freq[t])

#   Calculating Term Frequency
TF = freq
for i in range(len(vocab)):
    for j in range(n):
        if freq[j][i] > 0:             # If Term Frequency > 0
            TF[j][i] = float(format(1 + math.log2(freq[j][i]), "0.3f"))
for t in range(n):
    print("\nTerm Frequency for", file_name[t], "\n", TF[t])

#   Calculating Inverse Document Frequency
IDF = []
for i in range(len(vocab)):
    count = 0
    for j in range(n):
        if TF[j][i] > 0:
            count += 1
    IDF.append(float(format(math.log2(n / count), "0.3f")))
print("\nInverse Document Frequencies\n", IDF)

# Calculating Weight of each Term in Vocabulary
WGT = freq
for j in range(n):
    for i in range(len(vocab)):
        WGT[j][i] = freq[j][i] * IDF[i]
print("\nWeights of Vocabulary Terms\n", WGT)

# Calculating Cosine Similarity between Documents
for i in range(len(file_name)):
    for j in range(len(file_name)):
        print(f'Cosine Similarity between {file_name[i]} & {file_name[j]} : {cos_sim(WGT[i], WGT[j])}')

# for t in range(n):
#     for m in range(t, n):
#         if m != t:
#             print(f'Cosine Similarity ({file_name[t]},{file_name[m]}) : {cos_sim(WGT[t], WGT[m])}')
