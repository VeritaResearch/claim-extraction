import pandas as pd
import numpy as np
import os
from os.path import join, dirname, pardir
from sklearn.model_selection import train_test_split


import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

from src.data_utils import process_AVeriTeC, process_Claimbuster, process_PoliClaim

RANDOM_SEED = 42
TEST_PERC = 0.2
SAVE_DATA = True
SAVE_PATH = '../data/ours'

if __name__ == "__main__":
    dfs = [process_PoliClaim(), process_AVeriTeC(), process_Claimbuster()]
    df = pd.concat(dfs, ignore_index=True)
    
    df = df.sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)

    train_n = int(np.floor(df.shape[0]*(1-TEST_PERC)))
    test_n = df.shape[0] - train_n
    print(train_n, test_n)

    train_df = df.head(train_n)
    test_df = df.tail(test_n)

    if SAVE_DATA:
        train_df.to_csv(join(SAVE_PATH, 'train.csv'), index=False)
        train_df.to_csv(join(SAVE_PATH, 'test.csv'), index=False)



    
