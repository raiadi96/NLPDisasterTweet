import pandas as pd
import os
import numpy as np



def load_data(path):
    if path != None or path != '':
        train_dir = os.path.join(path, 'train.csv')
        test_dir = os.path.join(path, 'test.csv')

        df_train = pd.read_csv(train_dir, dtype={'id': np.int16, 'target': np.int8})
        df_test = pd.read_csv(test_dir, dtype={'id': np.int16})
        return df_train, df_test
    else:
        return None, None