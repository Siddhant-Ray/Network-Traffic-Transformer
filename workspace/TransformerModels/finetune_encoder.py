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
from utils import vectorize_features_to_numpy_finetune, vectorize_features_to_numpy_finetune_memento
from utils import sliding_window_features, sliding_window_delay
from utils import PacketDataset, gelu
from utils import PacketDatasetEncoder

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

# Hyper parameters from config file

with open('configs/config-encoder-test.yaml') as f:
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
class TransformerEncoderFinetune(pl.LightningModule):

    def __init__(self,input_size, loss_function):
        super(TransformerEncoderFinetune, self).__init__()

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
        X, y = train_batch
        self.lr_update()               
        prediction = self.forward(X)
        loss = self.loss_func(prediction, y)
        self.log('Train loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch
        prediction = self.forward(X)
        loss = self.loss_func(prediction, y)
        self.log('Val loss', loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y  = test_batch
        prediction = self.forward(X)
        loss = self.loss_func(prediction, y)
        self.log('Test loss', loss, sync_dist=True)
        return loss

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def training_epoch_end(self,outputs):
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log('Avg loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)


def main():
    path = "congestion_1/"
    files = ["endtoenddelay_test.csv"]

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    num_features = 15 + 1 #(Dummy delay added to maintain shape)
    input_size = sl_win_size * num_features
    output_size = sl_win_size

    # Choose fine-tuning dataset
    MEMENTO = False

    if MEMENTO:
        path = "memento_data/"
        files = ["memento_test10_final.csv"]

    else:
        path = "congestion_1/"
        files = ["endtoenddelay_test.csv"]


    model = TransformerEncoderFinetune(input_size, LOSSFUNCTION)
    '''cpath = "encoder_masked_logs2/pretrained_window40.ckpt"
    model = TransformerEncoderFinetune.load_from_checkpoint(input_size = input_size, loss_function = LOSSFUNCTION, checkpoint_path=cpath,
                                                            strict=False)'''

    ## Add a new classifier head for delay prediction                                                        
    model.decoderpred = nn.Sequential(model.decoderpred,
                                      nn.ReLU(),
                                      nn.Linear(input_size, output_size))  
                                                  
    full_feature_arr = []
    full_target_arr = []
    test_loaders = []

    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path+file)
        df = convert_to_relative_timestamp(df)
        df = ipaddress_to_number(df)

        if MEMENTO:
            feature_df, label_df = vectorize_features_to_numpy_finetune_memento(df)
        else:
            feature_df, label_df = vectorize_features_to_numpy_finetune(df)

        print(feature_df.head(), feature_df.shape)
        print(label_df.head())
        
        feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
        target_arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
        print(len(feature_arr), len(target_arr))
        full_feature_arr = full_feature_arr + feature_arr
        full_target_arr = full_target_arr + target_arr

    print(len(full_feature_arr), len(full_target_arr))
    exit()
    
    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(full_feature_arr, full_target_arr, test_size = 0.05,
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

    print("Started training at:")
    time = datetime.now()
    print(time)

    print("Removing old logs:")
    os.system("rm -rf finetune_encoder_logs/lightning_logs/")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="finetune_encoder_logs/")
    
    if NUM_GPUS >= 1:
        trainer = pl.Trainer(precision=16, gpus=-1, strategy="dp", max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])
    else:
        trainer = pl.Trainer(gpus=None, max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])

    trainer.fit(model, train_loader, val_loader)    
    print("Finished training at:")
    time = datetime.now()
    print(time)
    trainer.save_checkpoint("finetune_encoder_logs/finetuned_pretrained_window40.ckpt")

    if SAVE_MODEL:
        torch.save(model, "finetune_encoder_logs/finetuned_encoder_pretrained.pt")


    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "finetune_encoder_logs/lightning_logs/version_0"
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



