import torch
from pytorch_lightning.callbacks import ProgressBar
from torch import nn, optim, einsum
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random, os, pathlib
from ipaddress import ip_address
import pandas as pd, numpy as np
import json
import pytorch_lightning as pl
from tqdm import tqdm
import copy

from utils import *

if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print("Number of GPUS: {}".format(NUM_GPUS))
else:
    print("ERROR: NO CUDA DEVICE FOUND")


class AbsPosEmb1DAISummer(nn.Module):
    """
    Given query q of shape [batch heads tokens dim] we multiply
    q by all the flattened absolute differences between tokens.
    Learned embedding representations are shared across heads
    """

    def __init__(self, tokens, dim_head):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: elements of the sequence
            dim_head: the size of the last dimension of q
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.abs_pos_emb = nn.Parameter(torch.randn(tokens, dim_head) * scale)

    def forward(self, q):
        return einsum('b h i d, j d -> b h i j', q, self.abs_pos_emb)



def main():
    path = "/local/home/sidray/packet_transformer/evaluations/congestion_1/"
    file = "endtoenddelay.csv"
    print(os.getcwd())

    df = get_data_from_csv(path+file)
    df = ipaddress_to_number(df)
    feature_df, label_df = vectorize_features_to_numpy(df)

    print(feature_df.head(), feature_df.shape)
    print(label_df.head())

    sl_win_start = 0 
    sl_win_size = 10
    sl_win_shift = 1

    arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
    print(len(arr))
    arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
    print(len(arr))
    print(arr[-1])
    
    
if __name__== '__main__':
    main()



