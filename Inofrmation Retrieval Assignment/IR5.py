import collections
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# Tokenization and Stemming of Words
def tokenize(text):
    tokens = word_tokenize(text)
    stems = [PorterStemmer().stem(item) for item in tokens]
    return stems


# Compute Confusion Matrix and Purity
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # purity
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


# Import Dataset
documents = pd.read_csv("bbc-text.csv")

# Label Encoding
labelEncoder = preprocessing.LabelEncoder()
documents['category'] = labelEncoder.fit_transform(documents['category'])

df = pd.DataFrame(list(zip(documents['text'], documents['category'])), columns=['text', 'label'])

# Creating TF/IDF Vectors from Text
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
x = tfidf_vectorizer.fit_transform(df.text.values)

# Clustering with Different K Values
for no_of_clusters in (3, 5, 7, 9):
    # Build the K Means Model
    model = KMeans(n_clusters=no_of_clusters)

    # Train the Model
    model.fit_transform(x)
    clusters = collections.defaultdict(list)
    for doc_id, label in enumerate(model.labels_):
        clusters[label].append(doc_id)
    purity = purity_score(y_true=df.label, y_pred=model.labels_)

    # Output: Performance Measures
    print("\nFor Number of Clusters = ", no_of_clusters)
    print("\n\tPurity: \t", purity)
    print("\tPrecision: \t", metrics.precision_score(y_pred=model.labels_, y_true=df.label, average='macro'))
    print("\tRecall: \t", metrics.recall_score(y_pred=model.labels_,y_true=df.label, average='macro'))
    print("\tF-score: \t", metrics.f1_score(y_pred=model.labels_,y_true=df.label, average='macro'))