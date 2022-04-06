from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import Word2Vec


class WordEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, model_path: str, agg_type: str = 'mean') -> None:
        super().__init__()
        
        self.tknzr = TweetTokenizer(preserve_case=False)
        self.model = Word2Vec.load(model_path)
        
        self.agg_type = agg_type
        self.model_path = model_path


    def process_text(self, text):
        tokens = self.tknzr.tokenize(text)
        words = [w for w in tokens if len(w) > 1]

        return words 

    def fit_transform(self, X, y=None, **fit_params):
        embedding = []

        for sentence in X:
            words = self.process_text(sentence)
            x = self.model.wv[words].mean(axis=0)
            
            embedding.append(x)
            
        return np.array(embedding)

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return self.fit_transform(X=X, y=y)
