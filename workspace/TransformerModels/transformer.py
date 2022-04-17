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
from torch.nn import functional as F

from sklearn.model_selection import train_test_split

from utils import get_data_from_csv, ipaddress_to_number, vectorize_features_to_numpy
from utils import sliding_window_features, sliding_window_delay
from utils import PacketDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

# Hyper parameters
# TODO: Move to YAML/TOML config later

WEIGHTDECAY = 1e-5
LEARNINGRATE = 1e-4            
DROPOUT = 0.2                   
NHEAD = 8                      
LAYERS = 2                      
EPOCHS = 50
BATCHSIZE = 64
LINEARSIZE = 256


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

class BaseTransformer(pl.LightningModule):

    def __init__(self,input_size, target_size):
        super(BaseTransformer, self).__init__()

        self.step = [0]
        self.warmup_steps = 1000

        # create the model with its layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=LINEARSIZE, nhead=NHEAD, batch_first=True, dropout=DROPOUT)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=LINEARSIZE, nhead=NHEAD, batch_first=True, dropout=DROPOUT)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=LAYERS)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=LAYERS)
        self.encoderin = nn.Linear(input_size, LINEARSIZE)
        self.decoderin = nn.Linear(target_size, LINEARSIZE)
        self.decoderpred= nn.Linear(LINEARSIZE, target_size)
        self.model = nn.ModuleList([self.encoder, self.decoder, self.encoderin, self.decoderin, self.decoderpred])

        self.loss_func = nn.MSELoss()
        parameters = {"WEIGHTDECAY": WEIGHTDECAY, "LEARNINGRATE": LEARNINGRATE, "EPOCHS": EPOCHS, "BATCHSIZE": BATCHSIZE,
                         "LINEARSIZE": LINEARSIZE, "NHEAD": NHEAD, "LAYERS": LAYERS}
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        return {"optimizer": self.optimizer}

    def lr_update(self):
        self.step[0] += 1
        learning_rate = LINEARSIZE ** (-0.5) * min(self.step[0] ** (-0.5), self.step[0] * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def forward(self, input, target):
        # used for the forward pass of the model
        scaled_input = self.encoderin(input.double())
        target = self.decoderin(target.double())
        enc = self.encoder(scaled_input)
        out = self.decoderpred(self.decoder(target, enc))
        return out

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        loss = 0
        self.lr_update()
        prediction = self.forward(X, y)
        loss = F.mse_loss(prediction, y)
        self.log('Train loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch
        loss = 0
        prediction = self.forward(X, y)
        loss = F.mse_loss(prediction, y)
        self.log('Val loss', loss)
        return loss


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

    print(train_vectors[0].shape[0])
    print(train_labels[0].shape[0])

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    # print(train_dataset.__getitem__(0))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    feature = train_features[0]
    label = train_labels[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    model = BaseTransformer(train_vectors[0].shape[0], train_labels[0].shape[0])

    trainer = pl.Trainer(gpus=0, max_epochs=EPOCHS, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)    
    
    
if __name__== '__main__':
    main()



