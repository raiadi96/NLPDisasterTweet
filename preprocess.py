import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from wordcloud import STOPWORDS
import numpy as np
import data_plot as dp

OUTPUT_PATH = r"reports"

def missing_values(df_train, df_test):
    missing_cols = ['keyword', 'location']
    fig, axes = plt.subplots(ncols = 2, figsize = (17,4), dpi = 100)
    sns.barplot(x = df_train[missing_cols].isnull().sum().index,\
                y = df_train[missing_cols].isnull().sum().values, ax = axes[0])
    sns.barplot(x = df_test[missing_cols].isnull().sum().index, \
                y = df_test[missing_cols].isnull().sum().values, ax = axes[1])

    axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)

    axes[0].set_title('Training Set', fontsize=13)
    axes[1].set_title('Test Set', fontsize=13)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file_name = "target-dist-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    plt.savefig(os.path.join(OUTPUT_PATH, file_name))
    plt.clf()
    plt.cla()

    for df in [df_train, df_test]:
        for col in ['keyword', 'location']:
            df[col] = df[col].fillna(f'no_{col}')

    return df_train, df_test


def unqiue_words(df_train, df_test):
    print(
        f'Number of unique values in keyword = {df_train["keyword"].nunique()} (Training) - {df_test["keyword"].nunique()} (Test)')
    print(
        f'Number of unique values in location = {df_train["location"].nunique()} (Training) - {df_test["location"].nunique()} (Test)')


def generate_meta_features(df_train: pd.DataFrame, df_test: pd.DataFrame):

    #word count
    df_train['word_count'] = df_train.text.apply(lambda x: len(str(x).split()))
    df_test['word_count'] = df_test.text.apply(lambda x: len(str(x).split()))

    #unique words
    df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
    df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

    #stop word count
    df_train["stop_word_count"] = df_train["text"].apply(\
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df_test["stop_word_count"] = df_test["text"].apply(\
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    #url_count
    df_train["url_count"] = df_train["text"].apply(lambda x: len([w for w in str(x).lower().split() \
                                                                  if 'http' in w or 'https' in w ]))
    df_test["url_count"] = df_test["text"].apply(lambda x: len([w for w in str(x).lower().split() \
                                                                  if 'http' in w or 'https' in w]))

    # mean_word_length
    df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    #char count
    df_train["char_count"] = df_train["text"].apply(lambda x: len(str(x)))
    df_test["char_count"] = df_test["text"].apply(lambda x: len(str(x)))

    #hashtag count
    df_train["hashtag_counts"] = df_train["text"].apply(lambda x: len([w for w in str(x) if '#' in w]))
    df_test["hashtag_counts"] = df_test["text"].apply(lambda x: len([w for w in str(x) if '#' in w]))

    df_train["mention_counts"] = df_train["text"].apply(lambda x: len([w for w in str(x) if '@' in w]))
    df_test["mention_counts"] = df_test["text"].apply(lambda x: len([w for w in str(x) if '@' in w]))

    dp.plot_meta_features(df_train, df_test)


def build_vocab(texts: pd.Series):
    tweets = texts.apply(lambda x: x.split()).values
    vocab = {}
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_embeddings_coverage(texts: pd.Series, embeddings):
    vocab = build_vocab(texts)
    covered = {}
    oov = {}
    n_covered = 0
    n_oov = 0

    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered +=vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
    vocab_coverage = len(covered)/len(vocab)
    return vocab_coverage
