import torch
from pytorch_lightning.callbacks import ProgressBar
from torch import nn, optim, einsum
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random, math
from ipaddress import ip_address
import pandas as pd, numpy as np
import json
import pytorch_lightning as pl
from tqdm import tqdm
import copy

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        feature = self.encodings[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.labels)

class PacketDatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        feature = self.encodings[idx]
        return feature

    def __len__(self):
        return len(self.encodings)

def gelu(x):
   return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_data_from_csv(input_file):
    df = pd.read_csv(input_file)
    return df 

def convert_to_relative_timestamp(df):
    t_arr = df['Timestamp'].iloc[0]
    t_arr_abs = df['Timestamp'] - t_arr
    df['Timestamp'] = t_arr_abs
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

def vectorize_features_to_numpy_masked(data_frame):
    feature_frame = data_frame.drop(['Packet ID', 'Interface ID'], axis = 1)
    # Shift the IP ID, ECN and the DSCP values by 1
    feature_frame["IP ID"] = feature_frame["IP ID"] + np.int64(1) 
    feature_frame["ECN"] = feature_frame["ECN"] + np.int64(1) 
    feature_frame["DSCP"] = feature_frame["DSCP"] + np.int64(1) 
    feature_frame["Delay"] = feature_frame["Delay"] # Scale to ms 
    label_frame = data_frame['Delay'] # Scale to ms 
    # Scale the timestamp to milli sec, to prevent masked confusion scale]
    # feature_frame["Timestamp"] = feature_frame["Timestamp"]*1000
    feature_frame['Combined'] = feature_frame.apply(lambda row: row.to_numpy(), axis=1)

    return feature_frame

def vectorize_features_to_numpy_finetune(data_frame):
    feature_frame = data_frame.drop(['Packet ID', 'Interface ID'], axis = 1)
    # Shift the IP ID, ECN and the DSCP values by 1
    feature_frame["IP ID"] = feature_frame["IP ID"] + np.int64(1) 
    feature_frame["ECN"] = feature_frame["ECN"] + np.int64(1) 
    feature_frame["DSCP"] = feature_frame["DSCP"] + np.int64(1) 
    label_frame = data_frame['Delay']
    feature_frame.drop(['Delay'], axis = 1, inplace=True)
    # For fine-tune, add all 0s as delays to maintain input shape
    feature_frame["Dummy Delay"] = np.int64(0) 
    # Scale the timestamp to milli sec, to prevent masked confusion scale]
    # feature_frame["Timestamp"] = feature_frame["Timestamp"]*1000
    feature_frame['Combined'] = feature_frame.apply(lambda row: row.to_numpy(), axis=1)

    return feature_frame, label_frame

def vectorize_features_to_numpy_memento(data_frame, reduced = False, normalize = True):
    feature_frame = data_frame.drop(['Packet ID','Workload ID', 'Application ID'], axis = 1)
    label_frame = data_frame['Delay'] # Scale to ms 
    feature_frame["Delay"] = feature_frame['Delay'] # Scale to ms 
    ### Keep the ddelay, mask nth delay in batch during training
    
    # feature_frame.drop(['Delay'], axis = 1, inplace=True)
    feature_frame['Combined'] = feature_frame.apply(lambda row: row.to_numpy(), axis=1)

    ## Only keep packet size and delay 
    if normalize:
        feature_frame_reduced = feature_frame[["Timestamp", "Normalised Packet Size", "Normalised Delay"]]
    else:
        feature_frame_reduced = feature_frame[["Timestamp", "Packet Size", "Delay"]]
    feature_frame_reduced["Combined"] = feature_frame_reduced.apply(lambda row: row.to_numpy(), axis=1)

    if reduced:
        return feature_frame_reduced, label_frame
    return feature_frame, label_frame

def vectorize_features_to_numpy_finetune_memento(data_frame, reduced = False, normalize = True):
    feature_frame = data_frame.drop(['Packet ID', 'Workload ID', 'Application ID'], axis = 1)
    # Shift the IP ID, ECN and the DSCP values by 1
    feature_frame["IP ID"] = feature_frame["IP ID"] + np.int64(1) 
    feature_frame["ECN"] = feature_frame["ECN"] + np.int64(1) 
    feature_frame["DSCP"] = feature_frame["DSCP"] + np.int64(1) 
    label_frame = data_frame['Delay'] # Scale to ms 
    feature_frame["Delay"] = feature_frame['Delay'] # Scale to ms 

    ### Keep the ddelay, mask nth delay in batch during training

    # feature_frame.drop(['Delay'], axis = 1, inplace=True)
    # For fine-tune, add all 0s as delays to maintain input shape
    # feature_frame["Dummy Delay"] = np.int64(0) 
    # Scale the timestamp to milli sec, to prevent masked confusion scale]
    # feature_frame["Timestamp"] = feature_frame["Timestamp"]*1000
    feature_frame['Combined'] = feature_frame.apply(lambda row: row.to_numpy(), axis=1)

    ## Only keep packet size and delay 
    if normalize:
        feature_frame_reduced = feature_frame[["Timestamp", "Normalised Packet Size", "Normalised Delay"]]
    else:
        feature_frame_reduced = feature_frame[["Timestamp", "Packet Size", "Delay"]]
    feature_frame_reduced["Combined"] = feature_frame_reduced.apply(lambda row: row.to_numpy(), axis=1)

    if reduced:
        return feature_frame_reduced, label_frame

    return feature_frame, label_frame

# Features for message completion time (message size and MCT)
def create_features_for_MCT(data_frame, reduced = True, normalize = True):
    feature_frame = data_frame
    label_frame = data_frame['Delay'] # Scale to ms 


    feature_frame_reduced = feature_frame[["Timestamp", "Delay", "Packet Size", "Application ID", "Message ID"]]
    feature_frame_size_MCT = feature_frame_reduced[["Packet Size", "Application ID", "Message ID"]].groupby(["Application ID", "Message ID"]).sum()
    feature_frame_size_MCT.rename(columns = {"Packet Size" :"Message Size"}, inplace = True, copy = False)

    feature_frame_reduced["Transmissions"] = feature_frame_reduced["Timestamp"]
    feature_frame_reduced["Receptions"] = feature_frame_reduced["Timestamp"] + feature_frame_reduced["Delay"]

    feature_frame_FT_MCT = feature_frame_reduced[["Transmissions", "Application ID", "Message ID"]].groupby(["Application ID", "Message ID"]).first()
    feature_frame_FT_MCT.rename(columns = {"Transmissions" :"Message Timestamp"}, inplace = True, copy = False)

    feature_frame_LT_MCT = feature_frame_reduced[["Receptions", "Application ID", "Message ID"]].groupby(["Application ID", "Message ID"]).last()
    feature_frame_LT_MCT.rename(columns = {"Receptions" :"Message Timestamp"}, inplace = True, copy = False)
    
    feature_frame_TT_MCT = feature_frame_LT_MCT - feature_frame_FT_MCT
    feature_frame_TT_MCT.rename(columns = {"Message Timestamp" :"Message Completion Time"}, inplace = True, copy = False)
    feature_frame_FT_MCT.rename(columns = {"Message Timestamp" : "Message Creation Timestamp"}, inplace = True, copy = False)

    
    if reduced:
        final_df = feature_frame_size_MCT.merge(feature_frame_TT_MCT,
                             on = ["Application ID", "Message ID"], how='inner')
        final_df = final_df.merge(feature_frame_FT_MCT,
                                 on = ["Application ID", "Message ID"], how='inner')   

    final_df["Log Message Size"] = np.log(final_df["Message Size"])
    final_df["Log Message Completion Time"] = np.log(final_df["Message Completion Time"])

    if normalize:
        mean_size = final_df["Log Message Size"].mean()
        std_size= final_df["Log Message Size"].std()

        mean_mct = final_df["Log Message Completion Time"].mean()
        std_mct = final_df["Log Message Completion Time"].std()

        final_df["Normalised Log Message Size"] = (final_df["Log Message Size"] - mean_size)/std_size
        final_df["Normalised Log MCT"] = (final_df["Log Message Completion Time"] - mean_mct)/std_mct

    list_of_arrays = []
    for message_ts in final_df["Message Creation Timestamp"]:  # timestamp when message was generated
        recent_packets_size = feature_frame[feature_frame['Timestamp'] < message_ts].size
        
        if recent_packets_size < 1024:
            final_df.drop(final_df[final_df["Message Creation Timestamp"] == message_ts].index, axis = 0, inplace = True)
            continue

        # Get recent 1024 packets (as that is our window size)
        recent_packets = feature_frame[feature_frame['Timestamp'] < message_ts].tail(1024)
        recent_packets_features = recent_packets[["Timestamp", "Normalised Packet Size", "Normalised Delay"]]
        array = np.vstack(recent_packets_features.values).flatten()
        
        list_of_arrays.append(array)

    final_df["Input"] = list_of_arrays

    return final_df, mean_mct, std_mct, mean_size, std_size
    

def sliding_window_features(df_series, start, size, step):
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

def sliding_window_delay(df_series, start, size, step):
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
                empty_arr = np.zeros(1)
                n_rem_arr = np.concatenate((n_rem_arr, empty_arr))
            
            # print(n_rem_arr.shape)
            final_arr.append(n_rem_arr)
            break
   
    return final_arr

### Faster sliding windows (MUCH faster)
def make_windows_features(input_array, window_length, num_features, batchsize):
    """Yield windows one by one."""
    # Adjust window and batch size for multiple features.
    _windowsize = num_features * window_length
    _batchsize = num_features * batchsize

    last_start = len(input_array) - _windowsize
    last_end = len(input_array) - _windowsize + 1
    for start_index in range(0, last_start, _batchsize):
        end_index = min(last_end, start_index + _batchsize)
        index_matrix = \
            np.arange(start_index, end_index, num_features)[:, None] + \
            np.arange(_windowsize)[None, :]

        # Yields batches of sequences as arrays
        # yield input_array[index_matrix]
        # Yields sequence after sequence
        yield from input_array[index_matrix]


def make_windows_delay(input_array, window_length, batchsize):
    """Yield windows one by one."""
    last_start = len(input_array) - window_length
    last_end = len(input_array) - window_length + 1
    for start_index in range(0, last_start, batchsize):
        end_index = min(last_end, start_index + batchsize) 
        index_matrix = np.arange(start_index, end_index)[:, None] + \
            np.arange(window_length)[None, :]
        
        # Yields batches of sequences as arrays
        # yield input_array[index_matrix]
        # Yields sequence after sequence
        yield from input_array[index_matrix]
