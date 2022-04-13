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

from sklearn.model_selection import train_test_split

from utils import get_data_from_csv, ipaddress_to_number, vectorize_features_to_numpy
from utils import sliding_window_features, sliding_window_delay
from utils import PacketDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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

    feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
    target_arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
    print(len(feature_arr), len(target_arr))

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(feature_arr, target_arr, test_size = 0.1,
                                                            shuffle = False)
    # print(len(train_vectors), len(train_labels))
    # print(len(val_vectors), len(val_labels))

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    # print(train_dataset.__getitem__(0))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    
if __name__== '__main__':
    main()



