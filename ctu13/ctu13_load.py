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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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

    # How to handle ports and ip?   how to handle hexports?

    # Handle NaNs           # NUMPY df['col'] = df['col'].replace(np.nan, 0)
    df['sTos'] = df['sTos'].fillna(0.0)             # sTos 0.0 most values
    df['dTos'] = df['sTos'].fillna(0.0)             # dTos 0.0 most values
    df.Dport[df.Dport.isnull()] = df['Sport']          # Dport to Sport
    df.Sport[df.Sport.isnull()] = df['Dport']           # Sport to Dport
    df.Dport[df.Dport.isnull()] = '55555'             # Handle the case when both Sport and Dport is 0
    df.Sport[df.Sport.isnull()] = '55555'             # Handle the case when both Sport and Dport is 0

    #df[df['Sport'].str.contains('0x')]
    #df['Sport2'] = int((df['Sport'], 0))
    #df['Dport2'] = int((df['Dport'], 0))
    #df['Sport'] = df['Sport2']
    #df['Dport'] = df['Dport2']

    df['Sport'] = df.apply(lambda row: ml_h.hex_to_int(row.Sport), axis=1)
    df['Dport'] = df.apply(lambda row: ml_h.hex_to_int(row.Dport), axis=1)

    # Cast types
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    # df[label].astype('int32').dtypes
    df['Sport'] = df['Sport'].astype('int32')
    df['Dport'] = df['Dport'].astype('int32')
    df['target'] = df['target'].astype('int32')
    df['Dport'] = df['Dport'].astype('int32')
    df['Dur'] = df['Dur'].astype('float32')
    df['sTos'] = df['sTos'].astype('float16')
    df['dTos'] = df['dTos'].astype('float16')

    # Bin values

    # One-hot encode values

    # Create new features

    # Create train, val, test 70,10,20 with the same percentage of cats in all
    y = df[label]

    # Drop columns
    df.drop(['Label'], axis='columns', inplace=True)

    train, test_val, train_y, test_val_y  = train_test_split(df, y, test_size=0.3, stratify=y)
    val, test, val_y, test_y = train_test_split(test_val, test_val_y, test_size=0.67, stratify=test_val_y)

    cols = list(train.columns)


    # Drop labels, where ToDo some targets?
    #train.drop(['Label'], axis='columns', inplace=True)
    #val.drop(['Label'], axis='columns', inplace=True)
    #test.drop(['Label'], axis='columns', inplace=True)

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
    N_ROWS = 77000         # Rows to get from each of the dataset files
    #N_ROWS = None
    CONCATENATE = True
    EXPLORE_FILE = 'ctu13_explore.html'
    LABEL = 'target'
    LABEL_FROM = 'Label'
    SUBSTRING = 'botnet'
    UPDATED_DATAFRAME = 'ctu13_all.csv'
    COLUMNS_TO_KEEP = ['StartTime','Dur','Proto','Dir','State','sTos','dTos','TotPkts','TotBytes','SrcBytes','target']     #   SrcAddr Sport DstAddr Dport

    times = []
    start_time = time.time()

    # Download CTU13 dataset
    #h.download_file(URL_CTU13, DATASET_BASEDIR)
    # Extract the data ToDo low prio

    df = ml_h.dataframes_load(DATASET_BASEDIR, FILE_EXTENSION, N_ROWS, CONCATENATE)

    # Create new target(label) column , (botnet, normal, background) I only use the first 2
    ml_h.df_column_create_contains_text(df, LABEL, LABEL_FROM, SUBSTRING)

    print(df.shape)
    #for col in df:
    #    print(df[col].value_counts())
    #ml_h.pandas_dataframe_describe(df)
    print(df[LABEL].value_counts())

    train, val, test, train_y, val_y, test_y = ctu_preparing(df, LABEL)

    ml_h.dataframe_save(train, os.path.join(DATASET_BASEDIR, 'train.csv'))
    ml_h.dataframe_save(val, os.path.join(DATASET_BASEDIR, 'val.csv'))
    ml_h.dataframe_save(test, os.path.join(DATASET_BASEDIR, 'test.csv'))
    ml_h.dataframe_save(train_y, os.path.join(DATASET_BASEDIR, 'train_y.csv'))
    ml_h.dataframe_save(val_y, os.path.join(DATASET_BASEDIR, 'val_y.csv'))
    ml_h.dataframe_save(test_y, os.path.join(DATASET_BASEDIR, 'test_y.csv'))


    #print(train.corr())

    #for col in df.columns:
    #    print(df[col].value_counts())

    print(train.dtypes)
    print(train.info())
    print(train.head())

    numerical = ['float64', 'float32', 'int64', 'int32', 'int16']
    df_num = df.select_dtypes(include=numerical)
    print(df_num.head())

    ml_h.dataframe_drop_columns(df, COLUMNS_TO_KEEP)


    matplotlib.use('Qt5Agg')
    df.hist(bins=100, figsize=(20, 15), log=True)
    #save_fig("attribute_histogram_plots")
    plt.show()


    # Explore dataframe
    #ml_h.dataframe_explore(df, EXPLORE_FILE, explorative=False, minimal=True)

    times.append(time.time())
    print('\n Seconds')
    for time0 in times:
        print(str(time0 - start_time))

