#!/usr/bin/env python3

###################
# ToDo
# load dataset
# prepare dataset
# describe dataset
# create model
# train model
#
#
#
##################

import numpy as np
import pandas as pd
import os
import ml_helper
import matplotlib as mp

# ctu13 malware netflow dataset
# loading and poreparing the dataset


# Find files in subdirs with correct extension
def files_in_dirs(base_dir, extension='.csv'):
    return_files = []
    for path, current_directory, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                return_files.append(os.path.join(path, file))
    return return_files

# Check if file exists
def file_exists(path):
    if os.path.exists(path):
        return True
    else:
        print('ERROR cannot find file ' + path)
        return False

# Load pandas datafram from data_path
def pandas_dataframe_load(data_path, n_rows=0):
    # check if file exists
    if not(os.path.exists(data_path)):
        print('ERROR File doesn\'t exist, aborting')
        exit(0)
        return false

    # Load data into pandas
    print('Loading dataset file into pandas frame : ' + data_path)
    if n_rows==0:
        data_csv = pd.read_csv(data_path)
    else:
        data_csv = pd.read_csv(data_path, n_rows=n_rows)

    print('Loaded dataset')

    return data_csv

# Load dataset into dataframes
def dataframes_load(base_dir, extension, n_rows=False, concatenate=True):
    # Find all dataset files with correct extension in any subfolder
    dataset_paths = files_in_dirs(base_dir, extension)

    df_return = []
    for dataset_path in dataset_paths:
        df_return.append(pandas_dataframe_load(dataset_path, n_rows))

    # Concatenate all datasets into one
    if concatenate:
        df_return = pd.concat(df_return)
    return df_return


# Pipeline to prepare dataset
def ctu_preparing(df):
    # Handle empty values

    # Bin values

    # One-hot encode values

    # Create new features

    # Create train, val, test 80,10,10 with the same percentage of cats in all

    #

    return df



if __name__ == "__main__":
    DATASET_BASEDIR = r'C:\ai\datasets\ctu13\CTU-13-Dataset'
    FILE_EXTENSION = '.binetflow'
    N_ROWS = 0
    CONCATENATE = True

    df = dataframes_load(DATASET_BASEDIR, FILE_EXTENSION, N_ROWS, CONCATENATE)

    # Check if df is a list of dfs
    if type(df) == list:
        for df_ in df:
            print(df_.shape)
    else:
        print('Not a list')

    exit(0)

    print('\ncolumns')
    print(df.columns)

    column = ''

    print(df['Label'].name)
    labels = df['Label'].unique()

    for label in labels:
        print(label)


    #ml_helper.pandas_dataframe_describe(df)

    #df_small = df.sample(n=1000)

    # Explore dataframe
    #ml_helper.dataframe_explore(df_small, dst_file, explorative=False, minimal=False)


