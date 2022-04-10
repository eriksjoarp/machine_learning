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
import wget
import helper as h
import ml_helper as ml_h
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import time

# ctu13 malware netflow dataset
# loading and preparing the dataset


# Pipeline to prepare dataset
def ctu_preparing(df, label):
    # Proto,  (['StartTime', 'Dur', 'Proto', 'SrcAddr', 'Sport', 'Dir', 'DstAddr','Dport', 'State', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes','Label'],
    # Proto,  (['StartTime', '', '', 'SrcAddr', 'Sport', , 'DstAddr','Dport',,, 'State'
    categorical = []    # 'Proto', 'Dir', 'sTos', 'dTos'
    numerical = []      # 'Dur', , 'TotPkts', 'TotBytes', 'SrcBytes',
    to_time = []
    bin = []

    # Cast types
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df[label].astype('int16').dtypes

    # How to handle ports and ip?

    # Handle empty values


    # Bin values

    # One-hot encode values

    # Create new features

    # Create train, val, test 70,10,20 with the same percentage of cats in all
    y = df[label]
    print(y.head())

    train, test_val, train_y, test_val_y  = train_test_split(df, y, test_size=0.3, stratify=y)
    val, test, val_y, test_y = train_test_split(test_val, test_val_y, test_size=0.67, stratify=test_val_y)

    # Drop labels, where ToDo some targets?
    train.drop(['Label'], axis='columns', inplace=True)
    val.drop(['Label'], axis='columns', inplace=True)
    test.drop(['Label'], axis='columns', inplace=True)

    print('\n\n ########       train val test      ######## \n')
    print(str(train.shape) + ' ' + str(val.shape) + ' ' + str(test.shape))
    print(train.head())
    print(val.head())
    print(test.head())
    print(train_y.head())
    print(val_y.head())
    print(test_y.head())

    return train, val, test, train_y, val_y, test_y


if __name__ == "__main__":
    DATASET_BASEDIR = r'C:\ai\datasets\ctu13\CTU-13-Dataset'
    URL_CTU13 = r'''https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2'''  # Download and extract the data into your DATASET_BASEDIR
    FILE_EXTENSION = '.binetflow'
    N_ROWS = 7700          # Rows to get from each of the dataset files
    #N_ROWS = None
    CONCATENATE = True
    EXPLORE_FILE = 'ctu13_explore.html'
    LABEL = 'target'
    LABEL_FROM = 'Label'
    SUBSTRING = 'botnet'

    times = []
    start_time = time.time()

    # Download CTU13 dataset
    #h.download_file(URL_CTU13, DATASET_BASEDIR)
    # Extract the data ToDo low prio

    df = ml_h.dataframes_load(DATASET_BASEDIR, FILE_EXTENSION, N_ROWS, CONCATENATE)

    # Create new target(label) column , (botnet, normal, background) I only use the first 2
    ml_h.df_column_create_contains_text(df, LABEL, LABEL_FROM, SUBSTRING)

    # Check if df is a list of dfs
    if type(df) == list:
        for df_ in df:
            print(df_.shape)
    else:
        print('Not a list')
        print(df.shape)

    for col in df:
        print(df[col].value_counts())

    ml_h.pandas_dataframe_describe(df)

    print(df[LABEL].value_counts())


    train, val, test, train_y, val_y, test_y = ctu_preparing(df, LABEL)

    times.append(time.time())
    print('\n Seconds')
    for time0 in times:
        print(str(time0 - start_time))


    '''
    matplotlib.use('Qt5Agg')
    df.hist(bins=100, figsize=(20, 15), log=True)
    #save_fig("attribute_histogram_plots")
    plt.show()
    '''

    # all records that contains NaN values
    #sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

    # Explore dataframe
    #ml_h.dataframe_explore(df, EXPLORE_FILE, explorative=False, minimal=True)


