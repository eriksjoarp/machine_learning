#!/usr/bin/env python3

###########################################
#
#   Help functionality for machine learning
#   Explore pandas dataset
#
#   Remove column from dataframe
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
import helper as h


# Load dataset into dataframes
def dataframes_load(base_dir, extension, n_rows=False, concatenate=True):
    # Find all dataset files with correct extension in any subfolder
    dataset_paths = h.files_in_dirs(base_dir, extension)

    df_return = []
    for dataset_path in dataset_paths:
        df_return.append(pandas_dataframe_load(dataset_path, n_rows))

    # Concatenate all datasets into one
    if concatenate:
        df_return = pd.concat(df_return)
    return df_return

# Load pandas datafram from data_path
def pandas_dataframe_load(data_path, n_rows=None):
    # check if file exists
    if not(os.path.exists(data_path)):
        print('ERROR File doesn\'t exist, aborting')
        exit(0)
        return False

    # Load data into pandas
    print('Loading dataset file into pandas frame : ' + data_path)
    data_csv = pd.read_csv(data_path, nrows=n_rows)

    '''if n_rows==0:
        data_csv = pd.read_csv(data_path)
    else:
        data_csv = pd.read_csv(data_path, n_rows=n_rows)'''

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
    print(df.info(show_counts=True, verbose=True))

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

def func(input):
    return input + '_erik'

# Check if string contains substring
def contains_substring(fullstring, substring, lowercase=True):
    if lowercase:
        substring = substring.lower()
        fullstring = fullstring.lower()

    if substring in fullstring:
        return 1
    else:
        return 0


# Create a new column based on text in another column, if it contains a substring it returns 1 otherwise 0
def df_column_create_contains_text(df, new_column_name , column_from , substring, lowercase = True):
    # Create new column
    #df[new_column_name] = df.apply(lambda row: h.contains_substring(row[column_from], column_contains_text, lowercase), axis=1)
    #df[new_column_name] = df.apply(lambda row : func(row[column_from]) , axis=1)
    df[new_column_name] = df.apply(lambda row: contains_substring(row[column_from], substring, lowercase), axis=1)
    #df_col_from = df[]



if __name__ == "__main__":

    dataframe_explore()
