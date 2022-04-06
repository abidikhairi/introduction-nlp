import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

from utils import WordEmbedding


logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    steps = [
        ('embedding', WordEmbedding(model_path='../data/sentiment140.bin', agg_type='mean')),
        ('classifier', SVC())
    ]

    pipeline = Pipeline(steps=steps)
    
    model = pipeline.fit(train_df['text'], y=train_df['target'])
    
    y_hat = model.predict(test_df['text'])
    
    accuracy = accuracy_score(test_df['target'], y_hat)
    cm = confusion_matrix(test_df['target'], y_hat)
    cmd = ConfusionMatrixDisplay(cm)
    
    print(f'accuracy: {accuracy * 100:.2f} %')
    
    cmd.plot()
    plt.show()
