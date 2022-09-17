# Orignal author: Siddhant Ray

import argparse
import copy
import itertools as it
import json
import math
import os
import pathlib
import random
import time as t
from datetime import datetime
from ipaddress import ip_address
from locale import normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from generate_sequences import generate_MTC_data, generate_sliding_windows
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import einsum, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (MCTDataset, PacketDataset, PacketDatasetEncoder,
                   convert_to_relative_timestamp, gelu, get_data_from_csv,
                   ipaddress_to_number, sliding_window_delay,
                   sliding_window_features, vectorize_features_to_numpy,
                   vectorize_features_to_numpy_memento)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)

# Hyper parameters from config file

with open("configs/config-encoder-test.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

WEIGHTDECAY = float(config["weight_decay"])
LEARNINGRATE = float(config["learning_rate"])
DROPOUT = float(config["dropout"])
NHEAD = int(config["num_heads"])
LAYERS = int(config["num_layers"])
EPOCHS = int(config["epochs"])
BATCHSIZE = int(config["batch_size"])
LINEARSIZE = int(config["linear_size"])

if "loss_function" in config.keys():
    if config["loss_function"] == "huber":
        LOSSFUNCTION = nn.HuberLoss()
    if config["loss_function"] == "smoothl1":
        LOSSFUNCTION = nn.SmoothL1Loss()
    if config["loss_function"] == "kldiv":
        LOSSFUNCTION = nn.KLDivLoss()

LOSSFUNCTION = nn.MSELoss()

# Params for the sliding window on the packet data
SLIDING_WINDOW_START = 0
SLIDING_WINDOW_STEP = 1
SLIDING_WINDOW_SIZE = 1024
WINDOW_BATCH_SIZE = 5000
PACKETS_PER_EMBEDDING = 25

TRAIN = True
PRETRAINED = True
SAVE_MODEL = True
MAKE_EPOCH_PLOT = False
TEST = True


def ewma(seq, alpha=1):
    w_new = alpha
    w_old = 1 - alpha

    output = [seq[0]]
    old_val = seq[0]
    for new_val in seq[1:]:
        old_val = w_new * new_val + w_old * old_val
        output.append(old_val)
    return np.array(output)


def mse(seq_a, seq_b):
    return np.mean((seq_a - seq_b) ** 2)


if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    print("Number of GPUS: {}".format(NUM_GPUS))
else:
    print("ERROR: NO CUDA DEVICE FOUND")
    NUM_GPUS = 0

# TRANSFOMER CLASS TO PREDICT DELAYS
class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size,
        target_size,
        loss_function,
        delay_mean,
        delay_std,
        packets_per_embedding,
        pool=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.step = [0]
        self.warmup_steps = 4000

        # create the model with its layers

        # These are our transformer layers (stay the same)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=LINEARSIZE, nhead=NHEAD, batch_first=True, dropout=DROPOUT
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=LAYERS)

        # This is our prediction layer, change for finetuning as needed

        self.norm1 = nn.LayerNorm(LINEARSIZE)
        self.linear1 = nn.Linear(LINEARSIZE, LINEARSIZE * 4)
        self.activ1 = nn.Tanh()
        self.norm2 = nn.LayerNorm(LINEARSIZE * 4)
        self.linear2 = nn.Linear(LINEARSIZE * 4, LINEARSIZE)
        self.activ2 = nn.GELU()
        self.encoderpred1 = nn.Linear(LINEARSIZE, input_size // 8)
        self.activ3 = nn.ReLU()
        self.encoderpred2 = nn.Linear(input_size // 8, target_size)

        self.loss_func = loss_function
        parameters = {
            "WEIGHTDECAY": WEIGHTDECAY,
            "LEARNINGRATE": LEARNINGRATE,
            "EPOCHS": EPOCHS,
            "BATCHSIZE": BATCHSIZE,
            "LINEARSIZE": LINEARSIZE,
            "NHEAD": NHEAD,
            "LAYERS": LAYERS,
        }
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

        ## Mask out the nth delay in every input sequence (do it at run time)
        self.input_size = input_size
        self.packet_size = int(self.input_size / SLIDING_WINDOW_SIZE)
        self.packets_per_embedding = packets_per_embedding

        # Change into hierarchical embedding for the encoder
        self.feature_transform1 = nn.Sequential(
            Rearrange(
                "b (seq feat) -> b seq feat",
                seq=SLIDING_WINDOW_SIZE,
                feat=self.packet_size,
            ),  # Make 1000
            nn.Linear(self.packet_size, LINEARSIZE),
            nn.LayerNorm(LINEARSIZE),  # pre-normalization
        )
        """self.feature_transform1 =  nn.Sequential(Rearrange('b (seq feat) -> b seq feat',
                            seq=SLIDING_WINDOW_SIZE // self.packets_per_embedding,
                                            feat=self.packet_size * self.packets_per_embedding), # Make 1008 size sequences to 48,                                
                            nn.Linear(self.packet_size  * self.packets_per_embedding, LINEARSIZE), # each embedding now has 21 packets
                            nn.LayerNorm(LINEARSIZE), # pre-normalization
                            )"""

        self.remaining_packets1 = SLIDING_WINDOW_SIZE - 32
        self.feature_transform2 = nn.Sequential(
            Rearrange("b (seq n) feat  -> b seq (feat n)", n=32),
            nn.Linear(LINEARSIZE * 32, LINEARSIZE),
            nn.LayerNorm(LINEARSIZE),  # pre-normalization
        )
        # self.feature_transform2 = nn.Identity()
        self.remaining_packets2 = (self.remaining_packets1 // 32) - 15
        self.feature_transform3 = nn.Sequential(
            Rearrange("b (seq n) feat -> b seq (feat n)", n=16),
            nn.Linear(LINEARSIZE * 16, LINEARSIZE),
            nn.LayerNorm(LINEARSIZE),  # pre-normalization
        )
        # self.feature_transform3 = nn.Identity()

        # Choose mean pooling
        self.pool = pool

        # Mean and std for the delay un-normalization
        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def configure_optimizers(self):
        # self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        # Regularise only the weights, not the biases (regularisation of biases is not recommended)

        weights_parameters = (
            p for name, p in self.named_parameters() if "bias" not in name
        )
        bias_parameters = (p for name, p in self.named_parameters() if "bias" in name)

        self.optimizer = optim.Adam(
            [
                {"params": weights_parameters, "weight_decay": WEIGHTDECAY},
                {"params": bias_parameters},
            ],
            betas=(0.9, 0.98),
            eps=1e-9,
            lr=LEARNINGRATE,
        )
        return {"optimizer": self.optimizer}

    def lr_update(self):
        self.step[0] += 1
        learning_rate = LINEARSIZE ** (-0.5) * min(
            self.step[0] ** (-0.5), self.step[0] * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def forward(self, _input):
        # used for the forward pass of the model

        # Cast to doubletensor
        scaled_input = _input.double()

        # Embed every packet to the embedding dimension
        scaled_input1 = self.feature_transform1(scaled_input)

        # Keep first 32, re-embed the rest
        scaled_input_final1 = scaled_input1[:, :32, :]
        scaled_input_embed1 = scaled_input1[:, 32:, :]

        # Embed seequences of 32 packets to the embedding dimension
        scaled_input_2 = self.feature_transform2(scaled_input_embed1)

        # Keep the first 15, re-embed the rest
        scaled_input_final2 = scaled_input_2[:, :15, :]
        scaled_input_embed2 = scaled_input_2[:, 15:, :]

        # Embed seequences of 16 packets to the embedding dimension
        scaled_input_3 = self.feature_transform3(scaled_input_embed2)
        scaled_input_final3 = scaled_input_3

        # Embedding the final input (stack along sequence dimension)
        final_input = torch.cat(
            (scaled_input_final1, scaled_input_final2, scaled_input_final3), dim=1
        )

        enc = self.encoder(final_input)

        if self.pool:
            enc1 = enc.mean(
                dim=1
            )  # DO MEAN POOLING for the OUTPUT (as every packet is projected to embedding)
        else:
            enc1 = enc[
                :, -1
            ]  # Take last hidden state (as done in BERT , in ViT they take first hidden state as cls token)

        out = enc1

        return out

    def training_step(self, train_batch, train_idx):
        X_feat, X_size, y = train_batch
        self.lr_update()

        # Every packet separately into the transformer (project to linear if needed)
        prediction = self.forward(X_feat)
        X_size = X_size.unsqueeze(1)
        mct_feature = torch.cat((prediction, X_size), dim=1)

        # Call the predictor on the combined feature for MCT
        mct_pred = self.predictor(mct_feature)

        # Compute the loss
        y = y.unsqueeze(1)
        loss = self.loss_func(mct_pred, y)
        self.log("Train loss", loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X_feat, X_size, y = val_batch

        prediction = self.forward(X_feat)
        X_size = X_size.unsqueeze(1)
        mct_feature = torch.cat((prediction, X_size), dim=1)

        # Call the predictor on the combined feature for MCT
        mct_pred = self.predictor(mct_feature)

        # Compute the loss
        y = y.unsqueeze(1)
        loss = self.loss_func(mct_pred, y)
        self.log("Val loss", loss)
        return loss

    def test_step(self, test_batch, test_idx):
        X_feat, X_size, y = test_batch

        prediction = self.forward(X_feat)
        X_size = X_size.unsqueeze(1)
        mct_feature = torch.cat((prediction, X_size), dim=1)

        # Call the predictor on the combined feature for MCT
        mct_pred = self.predictor(mct_feature)

        # Compute the loss
        y = y.unsqueeze(1)
        loss = self.loss_func(mct_pred, y)
        self.log("Test loss", loss)

        predictions = mct_pred
        actual_vals = y

        return {"Test Loss": loss, "predictions": predictions, "actuals": actual_vals}

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def training_epoch_end(self, outputs):
        loss_tensor_list = [item["loss"].to("cpu").numpy() for item in outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log(
            "Avg loss per epoch",
            np.mean(np.array(loss_tensor_list)),
            on_step=False,
            on_epoch=True,
        )

    def test_epoch_end(self, outputs):
        last_predicted_times = []
        last_actual_times = []

        for output in outputs:
            preds = list(
                output["predictions"].cpu().detach().numpy()
            )  # predicted last times
            labels = list(output["actuals"].cpu().detach().numpy())  # actual last times

        last_predicted_times.extend(preds)
        last_actual_times.extend(labels)

        print()
        print(
            "Check lengths for all as sanity ",
            len(last_predicted_times),
            len(last_actual_times),
        )

        # Compute 99%ile SE

        losses_array = np.square(np.subtract(last_predicted_times, last_actual_times))
        print("99%ile SE on MCT is ", np.percentile(losses_array, 99))


def main():

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    num_features = 3  # If only timestamp, packet size and delay, else 16
    input_size = sl_win_size * num_features
    output_size = 1

    full_feature_arr = []
    full_target_arr = []

    ## Get the data
    final_df, mean_delay, std_delay, mean_mct, std_mct = generate_MTC_data()

    if PRETRAINED:
        ## Model definition with delay scaling params (from pretrained model)
        cpath = "checkpoints/finetune_nonpretrained_window{}.ckpt".format(
            SLIDING_WINDOW_SIZE
        )
        model = TransformerEncoder.load_from_checkpoint(
            input_size=input_size,
            target_size=output_size,
            loss_function=LOSSFUNCTION,
            delay_mean=mean_delay,
            delay_std=std_delay,
            packets_per_embedding=PACKETS_PER_EMBEDDING,
            pool=True,
            checkpoint_path=cpath,
            strict=True,
        )
        # Freeze everything!! Try with non-freezing also

        """for params in model.parameters(): 
            params.requires_grad = False"""

    else:
        # Do not freeeze anything for non pre-trained
        model = TransformerEncoder(
            input_size,
            output_size,
            LOSSFUNCTION,
            mean_delay,
            std_delay,
            PACKETS_PER_EMBEDDING,
            pool=True,
        )

    ## Freeze the previous head (as it is not being used anyway)
    for params in model.norm1.parameters():
        params.requires_grad = False
    for params in model.norm2.parameters():
        params.requires_grad = False
    for params in model.encoderpred1.parameters():
        params.requires_grad = False
    for params in model.encoderpred2.parameters():
        params.requires_grad = False
    for params in model.linear1.parameters():
        params.requires_grad = False
    for params in model.linear2.parameters():
        params.requires_grad = False
    for params in model.activ1.parameters():
        params.requires_grad = False
    for params in model.activ2.parameters():
        params.requires_grad = False
    for params in model.activ3.parameters():
        params.requires_grad = False

    # New predictor head for MCT task
    model.predictor = nn.Sequential(
        nn.LayerNorm(LINEARSIZE + 1),
        nn.Tanh(),
        nn.Linear(LINEARSIZE + 1, LINEARSIZE * 4),
        nn.LayerNorm(LINEARSIZE * 4),
        nn.GELU(),
        nn.Linear(LINEARSIZE * 4, LINEARSIZE),
        nn.LayerNorm(LINEARSIZE),
        nn.Linear(LINEARSIZE, input_size // 8),
        nn.ReLU(),
        nn.Linear(input_size // 8, output_size),
    )

    for params in model.predictor.parameters():
        params.requires_grad = True

    full_features_packets = final_df["Input"]
    full_features_size = final_df["Normalised Log Message Size"]
    full_labels = final_df["Normalised Log MCT"]

    full_feature_arr = list(
        zip(full_features_packets.to_list(), full_features_size.to_list())
    )
    full_target_arr = full_labels.to_list()

    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(
        full_feature_arr, full_target_arr, test_size=0.05, shuffle=True, random_state=42
    )
    # print(len(full_train_vectors), len(full_train_labels))
    # print(len(test_vectors), len(test_labels))

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(
        full_train_vectors, full_train_labels, test_size=0.1, shuffle=False
    )
    # print(len(train_vectors), len(train_labels))
    # print(len(val_vectors), len(val_labels))

    # print(train_vectors[0].shape[0])
    # print(train_labels[0].shape[0])

    ## Naive MCT baseline on previous value of MCT

    MCTS = np.array(test_labels)

    targets = MCTS[1:]  # No previous MCT for first one
    predictions = MCTS[:-1]

    smoothed_001 = ewma(predictions, alpha=0.01)  # 0.01 smoothing
    smoothed_01 = ewma(predictions, alpha=0.1)  # 0.1 smoothing

    mse_001 = mse(targets, smoothed_001)
    mse_01 = mse(targets, smoothed_01)

    print("Alpha 0.01", mse_001)
    print("Alpha 0.1", mse_01)

    baseline_mse = np.mean(np.square(np.subtract(targets, predictions)))
    print("MSE on previous MCT as prediction is", baseline_mse)

    train_dataset = MCTDataset(train_vectors, train_labels)
    val_dataset = MCTDataset(val_vectors, val_labels)
    test_dataset = MCTDataset(test_vectors, test_labels)
    # print(train_dataset.__getitem__(0))
    # print(val_dataset.__getitem__(1))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )

    # Make new dir for storing logs
    os.system("mkdir -p finetune_mct_logs/")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="finetune_mct_logs/")

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=16,
            gpus=-1,
            strategy="dp",
            max_epochs=EPOCHS * 3,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )
    else:
        trainer = pl.Trainer(
            gpus=None,
            max_epochs=EPOCHS,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )

    if TRAIN:
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf finetune_mct_logs/lightning_logs/")

        trainer.fit(model, train_loader, val_loader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint(
            "finetune_mct_logs/finetune_pretrained_window{}.ckpt".format(
                SLIDING_WINDOW_SIZE
            )
        )

    if SAVE_MODEL:
        pass
        # torch.save(model, "encoder_delay_logs/finetuned_encoder_scratch.pt")

    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "finetune_mct_logs/lightning_logs/version_0"
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
            trainer.test(model, dataloaders=test_loader)
        else:

            model.eval()

            cpath = "finetune_mct_logs/finetune_pretrained_window{}.ckpt".format(
                SLIDING_WINDOW_SIZE
            )
            testmodel = TransformerEncoder.load_from_checkpoint(
                input_size=input_size,
                target_size=output_size,
                loss_function=LOSSFUNCTION,
                delay_mean=mean_delay,
                delay_std=std_delay,
                packets_per_embedding=PACKETS_PER_EMBEDDING,
                pool=False,
                checkpoint_path=cpath,
                strict=True,
            )

            trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
