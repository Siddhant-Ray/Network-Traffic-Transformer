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

from utils import get_data_from_csv, convert_to_relative_timestamp, ipaddress_to_number
from utils import vectorize_features_to_numpy, vectorize_features_to_numpy_masked
from utils import sliding_window_features, sliding_window_delay
from utils import PacketDataset, gelu
from utils import PacketDatasetEncoder

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

# Hyper parameters from config file

with open('configs/config-encoder.yaml') as f:
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
SLIDING_WINDOW_SIZE = 40

SAVE_MODEL = True
MAKE_EPOCH_PLOT = False
TEST = True

if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print("Number of GPUS: {}".format(NUM_GPUS))
else:
    print("ERROR: NO CUDA DEVICE FOUND")
    NUM_GPUS = 0 

# TRANSFOMER CLASS TO PREDICT DELAYS
class MaskedTransformerEncoder(pl.LightningModule):

    def __init__(self,input_size, loss_function, src_mask=False):
        super(MaskedTransformerEncoder, self).__init__()

        self.step = [0]
        self.warmup_steps = 1000

        # create the model with its layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=LINEARSIZE, nhead=NHEAD, batch_first=True, dropout=DROPOUT)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=LAYERS)
        self.encoderin = nn.Linear(input_size, LINEARSIZE)
        self.linear1 = nn.Linear(LINEARSIZE, LINEARSIZE*4)
        self.activ1 = nn.Tanh()
        self.linear2 = nn.Linear(LINEARSIZE*4, LINEARSIZE*4)
        self.activ2 = nn.GELU()
        self.norm = nn.LayerNorm(LINEARSIZE*4)
        self.decoderpred= nn.Linear(LINEARSIZE*4, input_size)

        self.loss_func = loss_function
        self.masked_loss_func = nn.CrossEntropyLoss()

        parameters = {"WEIGHTDECAY": WEIGHTDECAY, "LEARNINGRATE": LEARNINGRATE, "EPOCHS": EPOCHS, "BATCHSIZE": BATCHSIZE,
                         "LINEARSIZE": LINEARSIZE, "NHEAD": NHEAD, "LAYERS": LAYERS}
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

        self.mask = src_mask
        self.input_size = input_size
        self.packet_size = int(self.input_size/ SLIDING_WINDOW_SIZE)
        self.start_mask_pos = np.arange(0, int(self.input_size), self.packet_size)
        self.feature_size = int(self.input_size / SLIDING_WINDOW_SIZE)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        return {"optimizer": self.optimizer}

    def lr_update(self):
        self.step[0] += 1
        learning_rate = LINEARSIZE ** (-0.5) * min(self.step[0] ** (-0.5), self.step[0] * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def forward(self, input):
        # used for the forward pass of the model
        scaled_input = self.encoderin(input.double())
        enc = self.encoder(scaled_input)
        out = self.linear1(self.activ1(enc))
        out = self.norm(self.linear2(self.activ2(out)))
        out = self.decoderpred(out)
        return out

    def training_step(self, train_batch, train_idx):
        X = train_batch
        self.lr_update()
        mask = self.mask

        if mask:
            batch_delay_position = self.start_mask_pos[1:] - np.ones(self.start_mask_pos[1:].shape)
            batch_delay_position.astype(int)
            batch_delay_position = np.concatenate([batch_delay_position, [batch_delay_position[-1]+self.packet_size]])
            # We mask 30% delay positions in every sequence (30% of N packets in the window have delay masked)
            mask_size = int(0.30*SLIDING_WINDOW_SIZE)
            masked_indices = [np.random.choice(batch_delay_position).astype(int) for i in range(mask_size)]
            # print(masked_indices)
            batch_mask_indices = masked_indices
            batch_mask = torch.tensor([0 for i in range(len(batch_mask_indices))], device=self.device)
            batch_mask = batch_mask.double() 

            correct_out = X[:, [batch_mask_indices]]   
            X[:, [batch_mask_indices]] = batch_mask   
             
            prediction = self.forward(X)
            masked_pred = prediction[:, [batch_mask_indices]]
            
            loss = self.loss_func(masked_pred, correct_out)
            self.log('Train loss', loss)
            return loss

    def validation_step(self, val_batch, val_idx):
        X = val_batch
        prediction = self.forward(X)
        loss = self.loss_func(prediction, X)
        self.log('Val loss', loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, test_idx):
        X = test_batch
        prediction = self.forward(X)
        loss = self.loss_func(prediction, X)
        self.log('Test loss', loss, sync_dist=True)
        return loss

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X = test_batch
        prediction = self.forward(X)
        return prediction

    def training_epoch_end(self,outputs):
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log('Avg loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)


def main():
    path = "congestion_1/"
    files = ["endtoenddelay500s_1.csv", "endtoenddelay500s_2.csv",
             "endtoenddelay500s_3.csv", "endtoenddelay500s_4.csv",
            "endtoenddelay500s_5.csv"]

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP


    num_features = 16 # Packet information + delay 
    input_size = sl_win_size * num_features
    output_size = sl_win_size

    model = MaskedTransformerEncoder(input_size, LOSSFUNCTION, src_mask=True)
    full_feature_arr = []
    full_target_arr = []
    test_loaders = []

    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path+file)
        df = convert_to_relative_timestamp(df)
        
        df = ipaddress_to_number(df)
        feature_df = vectorize_features_to_numpy_masked(df)

        print(feature_df.head(), feature_df.shape)
        
        feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
        full_feature_arr = full_feature_arr + feature_arr

    print(len(full_feature_arr))
    
    full_train_vectors, test_vectors = train_test_split(full_feature_arr, test_size = 0.05,
                                                            shuffle = True, random_state=42)
    # print(len(full_train_vectors), len(full_train_labels))
    # print(len(test_vectors), len(test_labels))

    train_vectors, val_vectors = train_test_split(full_train_vectors, test_size = 0.1,
                                                            shuffle = False)
    # print(len(train_vectors), len(train_labels))
    # print(len(val_vectors), len(val_labels))

    # print(train_vectors[0].shape[0])
    # print(train_labels[0].shape[0])

    train_dataset = PacketDatasetEncoder(train_vectors)
    val_dataset = PacketDatasetEncoder(val_vectors)
    test_dataset = PacketDatasetEncoder(test_vectors)
    # print(train_dataset.__getitem__(0))

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers = 4)

    # print one dataloader item!!!!
    train_features = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    feature = train_features[0]
    print(f"Feature: {feature}")
    

    val_features = next(iter(val_loader))
    print(f"Feature batch shape: {val_features.size()}")
    feature = val_features[0]
    print(f"Feature: {feature}")

    test_features = next(iter(test_loader))
    print(f"Feature batch shape: {test_features.size()}")
    feature = test_features[0]
    print(f"Feature: {feature}")

    print("Started training at:")
    time = datetime.now()
    print(time)

    print("Removing old logs:")
    os.system("rm -rf encoder_masked_logs2/lightning_logs/")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="encoder_masked_logs2/")
    
    if NUM_GPUS >= 1:
        trainer = pl.Trainer(precision=16, gpus=-1, strategy="dp", max_epochs=EPOCHS, check_val_every_n_epoch=10,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=15)])
    else:
        trainer = pl.Trainer(gpus=None, max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=15)])

    trainer.fit(model, train_loader, val_loader)    
    print("Finished training at:")
    time = datetime.now()
    print(time)
    ## Manually save checkpoint as auto saving is getting a bit messy
    trainer.save_checkpoint("encoder_masked_logs2/pretrained_window40.ckpt")

    if SAVE_MODEL:
        torch.save(model, "encoder_masked_logs2/pretrained_encoder.pt")

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
        trainer.test(model, dataloaders = test_loader)


if __name__== '__main__':
    main()



