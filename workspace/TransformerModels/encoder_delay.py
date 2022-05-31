from locale import normalize
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
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils import get_data_from_csv, convert_to_relative_timestamp, ipaddress_to_number, vectorize_features_to_numpy
from utils import vectorize_features_to_numpy_memento
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

TRAIN = True
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
class TransformerEncoder(pl.LightningModule):

    def __init__(self,input_size, target_size, loss_function, delay_mean, delay_std):
        super(TransformerEncoder, self).__init__()

        self.step = [0]
        self.warmup_steps = 4000

        # create the model with its layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=LINEARSIZE,
                                                        nhead=NHEAD, batch_first=True, dropout=DROPOUT)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=LAYERS)
        # self.encoderin = nn.Linear(input_size, LINEARSIZE) Remove projection temporarily
        self.norm1 = nn.LayerNorm(LINEARSIZE)
        self.linear1 = nn.Linear(LINEARSIZE, LINEARSIZE*4)
        self.activ1 = nn.Tanh()
        self.norm2 = nn.LayerNorm(LINEARSIZE*4)
        self.linear2 = nn.Linear(LINEARSIZE*4, LINEARSIZE)
        self.activ2 = nn.GELU()
        self.encoderpred1= nn.Linear(LINEARSIZE, input_size // 8)
        self.activ3 = nn.ReLU()
        self.encoderpred2= nn.Linear(input_size // 8, target_size)

        self.loss_func = loss_function
        parameters = {"WEIGHTDECAY": WEIGHTDECAY, "LEARNINGRATE": LEARNINGRATE, "EPOCHS": EPOCHS, "BATCHSIZE": BATCHSIZE,
                         "LINEARSIZE": LINEARSIZE, "NHEAD": NHEAD, "LAYERS": LAYERS}
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]
        
        ## Mask out the nth delay in every input sequence (do it at run time)
        self.input_size = input_size
        self.packet_size = int(self.input_size/ SLIDING_WINDOW_SIZE)

        # Change into per packet embedding for the encoder
        self.transform =  nn.Sequential(Rearrange('b (seq feat) -> b seq feat',
                            seq=SLIDING_WINDOW_SIZE, feat=self.packet_size),
                            nn.Linear(self.packet_size, LINEARSIZE),
                            nn.LayerNorm(LINEARSIZE), # pre-normalization
                            )
        # Choose mean pooling
        self.pool = False

        # Mean and std for the delay un-normalization
        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        return {"optimizer": self.optimizer}

    def lr_update(self):
        self.step[0] += 1
        learning_rate = LINEARSIZE ** (-0.5) * min(self.step[0] ** (-0.5), self.step[0] * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def forward(self, _input):
        # used for the forward pass of the model
        scaled_input = _input.double()
        scaled_input = self.transform(scaled_input)
        enc = self.encoder(scaled_input)

        if self.pool:                
            enc1 = enc.mean(dim=1) # DO MEAN POOLING for the OUTPUT (as every packet is projected to embedding)
        else:
            enc1 = enc[:,-1] # Take last hidden state (as done in BERT , in ViT they take first hidden state as cls token)
        
        enc1 = self.norm1(enc1)
        out = self.norm2(self.linear1(self.activ1(enc1)))
        out = self.norm1(self.linear2(self.activ2(out)))
        out = self.encoderpred2(self.activ3(self.encoderpred1(out)))
        return out

    def training_step(self, train_batch, train_idx):
        X, y = train_batch
        self.lr_update()    
        
        # Mask our the nth packet delay delay, which is at position seq_len - 1  (640 is sequence length)
        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor([0.0], dtype = torch.double, requires_grad = True, device=self.device)
        batch_mask = batch_mask.double() 
        X[:, [batch_mask_index]] = batch_mask 

        # Every packet separately into the transformer (project to linear if needed)
        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean
        
        loss = self.loss_func(prediction, y[:,[SLIDING_WINDOW_SIZE-1]])
        self.log('Train loss', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor([0.0], dtype = torch.double, requires_grad = False, device=self.device)
        batch_mask = batch_mask.double() 
        X[:, [batch_mask_index]] = batch_mask 

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        loss = self.loss_func(prediction, y[:,[SLIDING_WINDOW_SIZE-1]])
        self.log('Val loss', loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y  = test_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor([0.0], dtype = torch.double, requires_grad = False, device=self.device)
        batch_mask = batch_mask.double() 
        X[:, [batch_mask_index]] = batch_mask 

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        loss = self.loss_func(prediction, y[:,[SLIDING_WINDOW_SIZE-1]])

        mse_loss = nn.MSELoss()
        target_size = SLIDING_WINDOW_SIZE
        last_delay_pos = target_size - 1 

        last_actual_delay = y[:,[last_delay_pos]]
        last_predicted_delay = prediction

        # Get fake prediction from mean of n-1 delays
        fake_prediction = torch.clone(y)
        fake_prediction = fake_prediction[:, :-1].mean(axis=1, keepdims=True)
        
        last_delay_loss = mse_loss(last_actual_delay, last_predicted_delay)
        
        self.log('Test loss', loss, sync_dist=True)
        return {"Test loss": loss, "last_delay_loss": last_delay_loss,
                 "last_predicted_delay": last_predicted_delay, "last_actual_delay": last_actual_delay,
                 "fake_predicted_delay": fake_prediction}

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def training_epoch_end(self,outputs):
        loss_tensor_list = [item['loss'].to('cpu').numpy() for item in outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log('Avg loss per epoch', np.mean(np.array(loss_tensor_list)), on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        last_delay_losses = []
        last_predicted_delay = []
        last_actual_delay = []
        fake_last_delay = []
        for output in outputs:
            last_packet_losses = list(output['last_delay_loss'].cpu().detach().numpy()) # Losses on last delay only 
            preds = list(output['last_predicted_delay'].cpu().detach().numpy()) # predicted last delays
            labels = list(output['last_actual_delay'].cpu().detach().numpy()) # actual last delays
            fakes = list(output['fake_predicted_delay'].cpu().detach().numpy()) # fake last delays 
            
            last_delay_losses.extend(last_packet_losses)
            last_predicted_delay.extend(preds)
            last_actual_delay.extend(labels)
            fake_last_delay.extend(fakes)

        print()
        print("Check lengths for all as sanity ", len(last_predicted_delay), len(last_actual_delay), len(fake_last_delay))
        
        print("Mean loss on last delay (averaged from batches) is : ", np.mean(np.array(last_delay_losses)))
        
        last_predicted_delay = np.array(last_predicted_delay)
        last_actual_delay = np.array(last_actual_delay)

        losses_array = np.square(np.subtract(last_predicted_delay, last_actual_delay))

        print("Mean loss on last delay (averaged from items) is : ", np.mean(losses_array))
        print("99%%ile loss is : ", np.quantile(losses_array, 0.99))

        fake_last_delay = np.array(fake_last_delay)
        fake_losses_array = np.square(np.subtract(fake_last_delay, last_actual_delay))

        print("Mean loss on ARMA predicted last delay (averaged from items) is : ", np.mean(fake_losses_array))
        print("99%%ile loss on ARMA predicted delay is : ", np.quantile(fake_losses_array, 0.99))

        save_path= "plot_values/3features/"
        np.save(save_path + "transformer_last_delay.npy", np.array(last_predicted_delay))
        np.save(save_path + "arma_last_delay.npy", np.array(fake_last_delay))
        np.save(save_path + "actual_last_delay.npy", np.array(last_actual_delay))

def main():

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    num_features = 3 # If only timestamp, packet size and delay, else 16
    input_size = sl_win_size * num_features
    output_size = 1

    full_feature_arr = []
    full_target_arr = []
    test_loaders = []

    # Choose fine-tuning dataset
    MEMENTO = True

    if MEMENTO:
        path = "memento_data/"
        files = ["topo_more_data_1_final.csv", "topo_more_data_2_final.csv" , "topo_more_data_3_final.csv",
                "topo_more_data_4_final.csv", "topo_more_data_5_final.csv", "topo_more_data_6_final.csv"]

    else:
        path = "congestion_1/"
        files = ["endtoenddelay_test.csv"]

    ## To calculate the global mean and std of the dataset
    global_df = pd.DataFrame(["Packet Size", "Delay"])
    for file in files:
        
        file_df = pd.read_csv(path+file)
        file_df = file_df[["Packet Size", "Delay"]]
        global_df = pd.concat([global_df, file_df], ignore_index=True)

    print(global_df.shape)
    mean_delay = global_df["Delay"].mean()
    std_delay = global_df["Delay"].std()
    mean_size = global_df["Packet Size"].mean()
    std_size = global_df["Packet Size"].std()
    
    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path+file)
        df = convert_to_relative_timestamp(df) 
        df = ipaddress_to_number(df)
        df["Normalised Delay"] = df["Delay"].apply(lambda x: (x - mean_delay)/std_delay)
        df["Normalised Packet Size"] = df["Packet Size"].apply(lambda x: (x - mean_size)/std_size)

        if MEMENTO:
            feature_df, label_df = vectorize_features_to_numpy_memento(df, reduced=True, normalize=True)
        else:
            feature_df, label_df = vectorize_features_to_numpy(df)

        print(feature_df.head(), feature_df.shape)
        print(label_df.head())
        
        feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
        target_arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
        print(len(feature_arr), len(target_arr))
        full_feature_arr = full_feature_arr + feature_arr
        full_target_arr = full_target_arr + target_arr

    print(len(full_feature_arr), len(full_target_arr))
    
    ## Model definition with delay scaling params
    model = TransformerEncoder(input_size, output_size, LOSSFUNCTION, mean_delay, std_delay)
    
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

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="encoder_delay_logs2/")
        
    if NUM_GPUS >= 1:
        trainer = pl.Trainer(precision=16, gpus=-1, strategy="dp", max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])
    else:
        trainer = pl.Trainer(gpus=None, max_epochs=EPOCHS, check_val_every_n_epoch=1,
                        logger = tb_logger, callbacks=[EarlyStopping(monitor="Val loss", patience=5)])

    if TRAIN:
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf encoder_delay_logs2/lightning_logs/")

        trainer.fit(model, train_loader, val_loader)    
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint("encoder_delay_logs2/finetune_nonpretrained_window40.ckpt")

    if SAVE_MODEL:
        pass 
        # torch.save(model, "encoder_delay_logs/finetuned_encoder_scratch.pt")

    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "encoder_delay_logs2/lightning_logs/version_0"
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
        if TRAIN:
            trainer.test(model, dataloaders = test_loader)
        else:
            cpath = "encoder_delay_logs2/finetune_nonpretrained_window40.ckpt"
            testmodel = TransformerEncoder.load_from_checkpoint(input_size = input_size, target_size = output_size,
                                                            loss_function = LOSSFUNCTION, delay_mean = mean_delay, 
                                                            delay_std = std_delay, checkpoint_path=cpath,
                                                            strict=True)
            testmodel.eval()
            trainer.test(testmodel, dataloaders = test_loader)


if __name__== '__main__':
    main()



