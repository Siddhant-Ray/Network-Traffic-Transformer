import torch
from pytorch_lightning.callbacks import ProgressBar
from torch import nn, optim, einsum
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
from ipaddress import ip_address
import pandas as pd, numpy as np
import json
import pytorch_lightning as pl
from tqdm import tqdm
import copy


def get_data_from_csv(input_file):
    df = pd.read_csv(input_file)
    return df 

def ipaddress_to_number(df):
    df['Source IP'] = df['Source IP'].apply(lambda s : int(ip_address(s.lstrip(" "))))
    df['Destination IP'] = df['Destination IP'].apply(lambda s : int(ip_address(s.lstrip(" "))))
    return df

def vectorize_features_to_numpy(data_frame):
    feature_frame = data_frame.drop(['Packet ID', 'Interface ID'], axis = 1)
    label_frame = data_frame['Delay']
    feature_frame.drop(['Delay'], axis = 1, inplace=True)
    feature_frame['Combined'] = feature_frame.apply(lambda row: row.to_numpy(), axis=1)

    return feature_frame, label_frame

def sliding_window(df_series, start, size, step):
    final_arr =[]
    pos = start
    assert(step <= size)

    while start < df_series.shape[0]:
        arr= []
        for value in range(pos, pos+size):
            # print(value)
           arr.append(df_series.iloc[value])

        pos += step
        start += 1
        narr = np.array(arr).flatten()
        # print(narr.shape)
        final_arr.append(narr)

        ## Treat the remaining features in the sliding window
        ## when the last sequence length is shorter , pad it with zeros
        if pos > df_series.shape[0] - size:
            rem_arr = []
            remain_size = df_series.shape[0] - pos
            for value in range(pos, pos+remain_size):
                rem_arr.append(df_series.iloc[value])
            n_rem_arr = np.array(rem_arr).flatten()
            # print(n_rem_arr.shape)

            excess_size = size - remain_size
            for iter in range(excess_size):
                empty_arr = np.zeros(df_series.iloc[value].shape[0])
                n_rem_arr = np.concatenate((n_rem_arr, empty_arr))
            
            # print(n_rem_arr.shape)
            final_arr.append(n_rem_arr)
            break
   
    return final_arr
