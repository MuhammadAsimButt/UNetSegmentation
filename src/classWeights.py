# This module computes class weights of a dataset if we have classes placed in directory stucture
from src import config
import sklearn
import numpy as np
import torch
import os
import pandas as pd

train_df  = pd.read_excel(config.TRAIN_CSV_PATH)
test_df  = pd.read_excel(config.TEST_CSV_PATH)

num_rows = train_df.shape[0]
print('Number of Rows in given dataframe : ', num_rows)

tot_rows = 0
num_rows = len(train_df[(train_df['NC']==1) & (train_df['G3']==0) & (train_df['G4']==0)])
tot_rows = tot_rows+num_rows
print('Number of Rows in dataframe in which NC ==1 : ', num_rows)
num_rows = len(train_df[(train_df['NC']==0) & (train_df['G3']==1) & (train_df['G4']==0)])
tot_rows = tot_rows+num_rows
print('Number of Rows in dataframe in which G3 ==1 : ', num_rows)
num_rows = len(train_df[(train_df['NC']==0) & (train_df['G3']==0) & (train_df['G4']==1)])
tot_rows = tot_rows+num_rows
print('Number of Rows in dataframe in which G4 ==1 : ', num_rows)
num_rows = len(train_df[(train_df['G5']==1) & (train_df['G3']==0) & (train_df['G4']==0)])
tot_rows = tot_rows+num_rows
print('Number of Rows in dataframe in which G5 ==1 : ', num_rows)
print('Total Number of Rows in dataframe  : ', tot_rows)


