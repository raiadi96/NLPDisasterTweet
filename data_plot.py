import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import datetime

OUTPUT_PATH = r"reports"

def plot_keywords_distribution(df_train:pd.DataFrame) -> bool:
    fig = plt.figure(figsize=(8,72), dpi=100)
    sns.countplot(y = df_train.sort_values(by='target_mean', ascending=False)['keyword'],\
                  hue = df_train.sort_values(by='target_mean', ascending=False)['target'],\
                  )
    plt.tick_params(axis = 'x', labelsize = 15)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend()
    plt.title("Target distribution in Keywords")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file_name = "target-dist-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    plt.savefig(os.path.join(OUTPUT_PATH, file_name))
    plt.clf()
    plt.cla()
    return True



def plot_meta_features(df_train:pd.DataFrame, df_test: pd.DataFrame) -> bool:
    META_FEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count'\
        , 'mean_word_length', 'char_count', 'hashtag_counts', 'mention_counts']

    DISASTER_TWEETS = df_train['target'] == 1

    fig, axes = plt.subplots(ncols=2, nrows=len(META_FEATURES), figsize=(20, 50), dpi=100)

    for i, feature in enumerate(META_FEATURES):
        sns.distplot(df_train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0], color='green')
        sns.distplot(df_train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='red')

        sns.distplot(df_train[feature], label='Training', ax=axes[i][1])
        sns.distplot(df_test[feature], label='Test', ax=axes[i][1])

        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].legend()

        axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file_name = "meta-{}".format(datetime.datetime.utcnow().strftime("%m%d%Y%H%M%S"))
    plt.savefig(os.path.join(OUTPUT_PATH, file_name))
    plt.clf()
    plt.cla()
    return True