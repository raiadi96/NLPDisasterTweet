SEED = 134
DATA_PATH = 'data'
EMBEDDING_PATH = 'Embeddings/glove.840B.300d.pkl'

import load_data
import preprocess
from data_plot import plot_keywords_distribution
import tensorflow_hub

df_train, df_test = load_data.load_data(DATA_PATH)
print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024 ** 2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024 ** 2))

print("Missing Values in the Train Dataset before pre processing. \n{}".format(df_train.isnull().sum()))
print("Missing Values in Test Dataset before pre processing \n{}".format(df_test.isnull().sum()))

df_train, df_test = preprocess.missing_values(df_train, df_test)

print("Missing Values in the Train Dataset after pre processing \n{}".format(df_train.isnull().sum()))
print("Missing Values in Test Dataset after pre processing \n{}".format(df_test.isnull().sum()))

preprocess.unqiue_words(df_train, df_test)

df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')
plot_keywords_distribution(df_train)
df_train.drop(columns = ['target_mean'], inplace = True)

preprocess.generate_meta_features(df_train, df_test)

#load glove embedding
embedding =  load_data.load_embedding(path=EMBEDDING_PATH)

train_vocab_coverage = preprocess.check_embeddings_coverage(df_train['text'], embeddings= embedding)
test_vocab_coverage = preprocess.check_embeddings_coverage(df_test['text'], embeddings=embedding)

print("Vocabulary Coverage in Training before preprocessing is {:.2f}%".format(train_vocab_coverage * 100))
print("Vocabulary Coverage in Testing before preprocessing is {:.2f}%".format(test_vocab_coverage * 100))

df_train['text_cleaned'] = df_train['text'].apply(\
    lambda s : preprocess.clean(s))
df_test['text_cleaned'] = df_test['text'].apply(\
    lambda s : preprocess.clean(s))


train_vocab_coverage = preprocess.check_embeddings_coverage(df_train['text_cleaned'], embeddings= embedding)
test_vocab_coverage = preprocess.check_embeddings_coverage(df_test['text_cleaned'], embeddings=embedding)

print("Vocabulary Coverage in Training before preprocessing is {:.2f}%".format(train_vocab_coverage * 100))
print("Vocabulary Coverage in Testing before preprocessing is {:.2f}%".format(test_vocab_coverage * 100))

bert = tensorflow_hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'\
                              , trainable=True)








