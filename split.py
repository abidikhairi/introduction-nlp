import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('./data/sentiment140.csv', names=['target', 'ids', 'date', 'flag', 'user', 'text'], encoding="ISO-8859-1")
    
    df.loc[df['target'] == 0, 'target'] = 0
    df.loc[df['target'] == 4, 'target'] = 1
    
    test_df = df.sample(frac=0.01, random_state=1234)
    train_df = df.sample(frac=0.02, random_state=1234)

    test_df.to_csv('./data/test.csv', index=False)
    train_df.to_csv('./data/train.csv', index=False)

    print('train dataset: ', train_df.shape)
    print('test dataset: ', test_df.shape)
