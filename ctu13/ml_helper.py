#!/usr/bin/env python3

###########################################
#
#   Help functionality for machine learning
#
#   Load dataset into pandas
#   Explore pandas dataset
#
#
#
#
#
#
#
#
#
#
###########################################

import os
import pandas
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import pandas_profiling

# Load dataset into pandas
def pandas_dataframe_load(data_path):
    # check if file exists
    if not(os.path.exists(data_path)):
        print('ERROR File doesn\'t exist, aborting')
        return false

    # read data into pandas
    print('Loading dataset file into pandas frame : ' + data_path)
    data_csv = pd.read_csv(data_path)
    print('Loaded dataset')

    return data_csv

def dataframe_explore(df, dst_file, title='Panda dataframe exploration' , explorative=False, minimal=False):
    print('Running pandas_profiling, explorative=' + str(explorative) + ' minimal=' + str(minimal) + ' report=' + dst_file)

    print(pandas_profiling.version)

    profile = ProfileReport(df, title=title, explorative=explorative, minimal=minimal)

    profile.to_file(dst_file)

def pandas_dataframe_describe(df):
    print('\ndf.shape')
    print(df.shape)

    print('\ninfo')
    print(df.info())

    print('\ndescribe')
    print(df.describe())

    print('\ncolumns')
    print(df.columns)

    print('\nhead')
    print(df.head())

    print('\ntail')
    print(df.tail())

    print('\ncorr')
    print(df.corr())

    print('\n')
    for col in df:
        print(df[col].name)
        print(df[col].value_counts())



if __name__ == "__main__":

    dataframe_explore(df)
