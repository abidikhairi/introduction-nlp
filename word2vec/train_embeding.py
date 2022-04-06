import logging
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer 


logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

def process_tweet(text):
    tknzr = TweetTokenizer(preserve_case=False)
    tokens = tknzr.tokenize(text)
    words = [w for w in tokens if len(w) > 1]
    
    return words


if __name__ == '__main__':
    data = pd.read_csv('../data/sentiment140.csv')

    sentences = list(map(process_tweet, data['text']))

    model = Word2Vec(sentences, vector_size=128, window=10, negative=5, sg=1, workers=6, epochs=5)
    model.save('../data/sentiment140.bin')
