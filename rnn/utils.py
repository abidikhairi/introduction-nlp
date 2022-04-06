import os
import json
import torch
from nltk.tokenize import TweetTokenizer


def prepare_sequence(sentence, word2idx):
    tknzr = TweetTokenizer(preserve_case=False)
    tokens = tknzr.tokenize(sentence)

    seq = [word2idx[token] for token in tokens if token in word2idx]

    return torch.tensor(seq, dtype=torch.long)


def load_or_build_vocab(datasets, vocab_path):
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            return json.load(f)

    tknzr = TweetTokenizer(preserve_case=False)
    vocab = {}

    for dataset in datasets:
        for sentence in dataset['text']:
            for word in tknzr.tokenize(sentence):
                if vocab.get(word, None) is None:
                    vocab[word] = len(vocab)
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)

    return vocab
