import random, os, pathlib
from ipaddress import ip_address
import pandas as pd, numpy as np
import json, copy, math
import yaml, time as t
from datetime import datetime

import argparse

from tqdm import tqdm

import torch
from pytorch_lightning.callbacks import ProgressBar
from torch import nn, optim, einsum
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils import get_data_from_csv, ipaddress_to_number, vectorize_features_to_numpy
from utils import sliding_window_features, sliding_window_delay
from utils import PacketDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

# Hyper parameters from config file

with open('configs/config-transformer.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

WEIGHTDECAY = float(config['weight_decay'])      
LEARNINGRATE = float(config['learning_rate'])         
DROPOUT = float(config['dropout'])                      
NHEAD = int(config['num_heads'])    
LAYERS = int(config['num_layers'])             
EPOCHS = int(config['epochs'])  
BATCHSIZE = int(config['batch_size'])  
LINEARSIZE = int(config['linear_size'])
LOSSFUNCTION = nn.MSELoss()

if 'loss_function' in config.keys():
    if config['loss_function'] == "huber":
        LOSSFUNCTION = nn.HuberLoss()
    if config['loss_function'] == "smoothl1":
        LOSSFUNCTION = nn.SmoothL1Loss()
    if config['loss_function'] == "kldiv":
        LOSSFUNCTION = nn.KLDivLoss()

# Params for the sliding window on the packet data 
SLIDING_WINDOW_START = 0
SLIDING_WINDOW_STEP = 1
SLIDING_WINDOW_SIZE = 10

SAVE_MODEL = False
MAKE_EPOCH_PLOT = True
TEST = True

if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print("Number of GPUS: {}".format(NUM_GPUS))
else:
    print("ERROR: NO CUDA DEVICE FOUND")
    NUM_GPUS = 0 

# DO NOT USE (AS OF NOW)
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

# DO NOT USE (AS OF NOW)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# TRANSFOMER CLASS TO PREDICT DELAYS
class BaseTransformer(pl.LightningModule):

    def __init__(self,input_size, target_size, loss_function):
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

        self.loss_func = loss_function
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

    def training_step(self, train_batch, train_idx):
        X, y = train_batch
        loss = 0
        self.lr_update()
        prediction = self.forward(X, y)
        loss = self.loss_func(prediction, y)
        self.log('Train loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch
        loss = 0
        prediction = self.forward(X, y)
        loss = self.loss_func(prediction, y)
        self.log('Val loss', loss)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y = test_batch
        loss = 0
        prediction = self.forward(X, y)
        loss = self.loss_func(prediction, y)
        self.log('Test loss', loss)
        return loss

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X, y = test_batch
        prediction = self.forward(X, y)
        return prediction

    def training_epoch_end(self,outputs):
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log('Avg loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)


def main():
    path = "/local/home/sidray/packet_transformer/evaluations/congestion_1/"
    file = "endtoenddelay.csv"
    print(os.getcwd())

    df = get_data_from_csv(path+file)
    df = ipaddress_to_number(df)
    feature_df, label_df = vectorize_features_to_numpy(df)

    print(feature_df.head(), feature_df.shape)
    print(label_df.head())

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
    target_arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
    print(len(feature_arr), len(target_arr))

    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(feature_arr, target_arr, test_size = 0.05,
                                                            shuffle = True, random_state=42)
    # print(len(full_train_vectors), len(full_train_labels))
    # print(len(test_vectors), len(test_labels))

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(full_train_vectors, full_train_labels, test_size = 0.1,
                                                            shuffle = False)
    # print(len(train_vectors), len(train_labels))
    # print(len(val_vectors), len(val_labels))

    # print(train_vectors[0].shape[0])
    # print(train_labels[0].shape[0])

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    test_dataset = PacketDataset(test_vectors, test_labels)
    # print(train_dataset.__getitem__(0))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers = 4)

    # print one dataloader item!!!!
    train_features, train_lbls = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_lbls.size()}")
    feature = train_features[0]
    label = train_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    val_features, val_lbls = next(iter(val_loader))
    print(f"Feature batch shape: {val_features.size()}")
    print(f"Labels batch shape: {val_lbls.size()}")
    feature = val_features[0]
    label = val_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    test_features, test_lbls = next(iter(test_loader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_lbls.size()}")
    feature = test_features[0]
    label = test_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    
    model = BaseTransformer(train_vectors[0].shape[0], train_labels[0].shape[0], LOSSFUNCTION)
    print("Started training at:")
    time = datetime.now()
    print(time)

    print("Removing old logs:")
    os.system("rm -rf transformer_logs/lightning_logs/")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="transformer_logs/")
    
    if NUM_GPUS > 1:
        trainer = pl.Trainer(precision=16, gpus=-1, strategy="dp", max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])
    else:
        trainer = pl.Trainer(gpus=None, max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])

    trainer.fit(model, train_loader, val_loader)    
    print("Finished training at:")
    time = datetime.now()
    print(time)

    if SAVE_MODEL:
        name = config['name']
        torch.save(model.model, f"./trained_transformer_{name}")

    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "transformer_logs/lightning_logs/version_0"
        y_key = "Avg loss per epoch"

        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        steps = {x.step for x in event_accumulator.Scalars("epoch")}
        epoch_vals = list({x.value for x in event_accumulator.Scalars("epoch")})
        epoch_vals.pop()

        x = list(range(len(steps)))
        y = [x.value for x in event_accumulator.Scalars(y_key) if x.step in steps]
        
        fig, ax = plt.subplots()
        ax.plot(epoch_vals, y)
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_key)
        fig.savefig("lossplot_perepoch.png")

    if TEST:
        trainer.test(model, dataloaders=test_loader)

if __name__== '__main__':
    main()



