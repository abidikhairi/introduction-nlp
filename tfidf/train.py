import os
import sys
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def train_lsi_features(data):
    tknzr = TweetTokenizer(preserve_case=False)
    
    vectorize = TfidfVectorizer(tokenizer=tknzr.tokenize)
    
    truncatedSVD = TruncatedSVD(n_components=128)

    features = vectorize.fit_transform(data['text'])
    
    latent_features = truncatedSVD.fit_transform(features)

    x_train, x_test, y_train, y_test = train_test_split(latent_features, data['target'], test_size=0.2)

    svc = SVC()

    model = svc.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    
    acc = accuracy_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)

    print(f"LSI + SVM classifier: {acc * 100:.2f}")

    cmd = ConfusionMatrixDisplay(cm)

    cmd.plot()
    plt.show()


def train_tfidf_features(data):
    tknzr = TweetTokenizer(preserve_case=False)
    
    vectorize = TfidfVectorizer(tokenizer=tknzr.tokenize)
    
    features = vectorize.fit_transform(data['text'])
    
    x_train, x_test, y_train, y_test = train_test_split(features, data['target'], test_size=0.2)

    svc = SVC()

    model = svc.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    
    acc = accuracy_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)

    print(f"TF-IDF + SVM classifier: {acc * 100:.2f}")

    cmd = ConfusionMatrixDisplay(cm)

    cmd.plot()
    plt.show()


if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    data = pd.concat([test_df, train_df])

    if sys.argv[1] == "tfidf":
        train_tfidf_features(data)
    elif sys.argv[1] == "lsi":
        train_lsi_features(data)
