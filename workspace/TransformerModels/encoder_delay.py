# Orignal author: Siddhant Ray

import argparse
import copy
import json
import math
import os, pickle
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
from generate_sequences import generate_sliding_windows
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import einsum, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    PacketDataset,
    PacketDatasetEncoder,
    convert_to_relative_timestamp,
    gelu,
    get_data_from_csv,
    ipaddress_to_number,
    sliding_window_delay,
    sliding_window_features,
    vectorize_features_to_numpy,
    vectorize_features_to_numpy_memento,
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)


# TRANSFOMER CLASS TO PREDICT DELAYS
class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size,
        target_size,
        loss_function1,
        loss_function2,
        delay_mean,
        delay_std,
        packets_per_embedding,
        layers,
        num_heads,
        dropout,
        weight_decay,
        learning_rate,
        epochs,
        batch_size,
        linear_size,
        sliding_window_size,
        dual_loss,
        mask_all_sizes,
        mask_all_delays,
        use_hierarchical_aggregation=True,
        pool=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.step = [0]
        self.warmup_steps = 4000

        self.layers = layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.linear_size = linear_size
        self.sliding_window_size = sliding_window_size
        self.dual_loss = dual_loss
        self.mask_all_sizes = mask_all_sizes
        self.mask_all_delays = mask_all_delays
        self.use_hierarchical_aggregation = use_hierarchical_aggregation

        # create the model with its layers
        # These are our transformer layers (stay the same)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.linear_size,
            nhead=self.num_heads,
            batch_first=True,
            dropout=self.dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers)

        # This is our prediction layer, change for finetuning as needed
        self.norm1 = nn.LayerNorm(self.linear_size)
        self.linear1 = nn.Linear(self.linear_size, self.linear_size * 4)
        self.activ1 = nn.Tanh()
        self.norm2 = nn.LayerNorm(self.linear_size * 4)
        self.linear2 = nn.Linear(self.linear_size * 4, self.linear_size)
        self.activ2 = nn.GELU()
        self.encoderpred1 = nn.Linear(self.linear_size, input_size // 8)
        self.activ3 = nn.ReLU()
        self.encoderpred2 = nn.Linear(input_size // 8, target_size)

        self.loss_func1 = loss_function1
        self.loss_func2 = loss_function2

        parameters = {
            "WEIGHTDECAY": self.weight_decay,
            "LEARNINGRATE": self.learning_rate,
            "EPOCHS": self.epochs,
            "BATCHSIZE": self.batch_size,
            "LINEARSIZE": self.linear_size,
            "NHEAD": self.num_heads,
            "LAYERS": self.layers,
        }
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

        ## Mask out the nth delay in every input sequence (do it at run time)
        self.input_size = input_size
        self.packet_size = int(self.input_size / self.sliding_window_size)
        self.packets_per_embedding = packets_per_embedding

        ## Use multi-level hierarchical aggregation
        if self.use_hierarchical_aggregation:
            # Change into hierarchical embedding for the encoder
            self.feature_transform1 = nn.Sequential(
                Rearrange(
                    "b (seq feat) -> b seq feat",
                    seq=self.sliding_window_size,
                    feat=self.packet_size,
                ),  # Make 1000
                nn.Linear(self.packet_size, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )
            self.remaining_packets1 = self.sliding_window_size - 32

            self.feature_transform2 = nn.Sequential(
                Rearrange("b (seq n) feat  -> b seq (feat n)", n=32),
                nn.Linear(self.linear_size * 32, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )
            self.remaining_packets2 = (self.remaining_packets1 // 32) - 15

            self.feature_transform3 = nn.Sequential(
                Rearrange("b (seq n) feat -> b seq (feat n)", n=16),
                nn.Linear(self.linear_size * 16, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )

        ## Use common size aggregation
        else:
            self.feature_transform1 = nn.Sequential(
                Rearrange(
                    "b (seq feat) -> b seq feat",
                    seq=self.sliding_window_size // self.packets_per_embedding,
                    feat=self.packet_size * self.packets_per_embedding,
                ),  # Make 1008 size sequences to 48,
                nn.Linear(
                    self.packet_size * self.packets_per_embedding, self.linear_size
                ),  # each embedding now has 21 packets
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )

            self.feature_transform2 = nn.Identity()
            self.feature_transform3 = nn.Identity()

        # Choose mean pooling
        self.pool = pool

        # Mean and std for the delay un-normalization
        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def configure_optimizers(self):
        # self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98),
        # eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        # Regularise only the weights, not the biases (regularisation of biases
        # is not recommended)
        weights_parameters = (
            p for name, p in self.named_parameters() if "bias" not in name
        )
        bias_parameters = (p for name, p in self.named_parameters() if "bias" in name)

        self.optimizer = optim.Adam(
            [
                {"params": weights_parameters, "weight_decay": self.weight_decay},
                {"params": bias_parameters},
            ],
            betas=(0.9, 0.98),
            eps=1e-9,
            lr=self.learning_rate,
            foreach=False,
        )
        return {"optimizer": self.optimizer}

    def lr_update(self):
        self.step[0] += 1
        learning_rate = self.linear_size ** (-0.5) * min(
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

        # Predict the output
        enc1 = self.norm1(enc1)
        out = self.norm2(self.linear1(self.activ1(enc1)))
        out = self.norm1(self.linear2(self.activ2(out)))
        out = self.encoderpred2(self.activ3(self.encoderpred1(out)))

        return out

    def training_step(self, train_batch, train_idx):
        X, y = train_batch
        self.lr_update()

        # Mask our the nth packet delay delay, which is at position seq_len - 1
        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=True, device=self.device
        )
        batch_mask = batch_mask.double()

        ## For ablation study, mask out all delays and all sizes
        batch_all_size_masks_index = np.arange(1, self.input_size, 3)
        batch_all_size_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_size_masks = batch_all_size_masks.double()

        batch_all_delay_masks_index = np.arange(2, self.input_size, 3)
        batch_all_delay_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_delay_masks = batch_all_delay_masks.double()

        X[:, [batch_mask_index]] = batch_mask

        if self.mask_all_sizes:
            ## Mask out all sizes
            X[:, batch_all_size_masks_index] = batch_all_size_masks
        if self.mask_all_delays:
            ## Mask out all delays
            X[:, batch_all_delay_masks_index] = batch_all_delay_masks

        # Add Gaussian noise to the input X for training
        # mean of noise is drawn from uniform distribution between
        # input range of X, std is 1e-5
        noise_mean = (torch.max(X) - torch.min(X)) / 2
        noise_std = 1e-5
        noise = torch.randn(X.size(), device=self.device) * noise_std + noise_mean
        # X = X + noise

        # Every packet separately into the transformer (project to linear if needed)
        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        ## Loss over both the loss functions
        loss_recnstr = self.loss_func1(prediction, y[:, [self.sliding_window_size - 1]])
        if self.dual_loss:
            loss_entropy = self.loss_func2(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            loss = loss_recnstr + loss_entropy
        else:
            loss = loss_recnstr
        self.log("Train loss", loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=False, device=self.device
        )
        batch_mask = batch_mask.double()

        ## For ablation study, mask out all delays and all sizes
        batch_all_size_masks_index = np.arange(1, self.input_size, 3)
        batch_all_size_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_size_masks = batch_all_size_masks.double()

        batch_all_delay_masks_index = np.arange(2, self.input_size, 3)
        batch_all_delay_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_delay_masks = batch_all_delay_masks.double()

        X[:, [batch_mask_index]] = batch_mask

        if self.mask_all_sizes:
            ## Mask out all sizes
            X[:, batch_all_size_masks_index] = batch_all_size_masks
        if self.mask_all_delays:
            ## Mask out all delays
            X[:, batch_all_delay_masks_index] = batch_all_delay_masks

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        ## Loss over both the loss functions
        loss_recnstr = self.loss_func1(prediction, y[:, [self.sliding_window_size - 1]])
        if self.dual_loss:
            loss_entropy = self.loss_func2(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            loss = loss_recnstr + loss_entropy
        else:
            loss = loss_recnstr
        self.log("Val loss", loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y = test_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=False, device=self.device
        )
        batch_mask = batch_mask.double()

        ## For ablation study, mask out all delays and all sizes
        batch_all_size_masks_index = np.arange(1, self.input_size, 3)
        batch_all_size_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_size_masks = batch_all_size_masks.double()

        batch_all_delay_masks_index = np.arange(2, self.input_size, 3)
        batch_all_delay_masks = torch.zeros(
            self.input_size // 3,
            dtype=torch.double,
            requires_grad=True,
            device=self.device,
        )
        batch_all_delay_masks = batch_all_delay_masks.double()

        X[:, [batch_mask_index]] = batch_mask

        if self.mask_all_sizes:
            ## Mask out all sizes
            X[:, batch_all_size_masks_index] = batch_all_size_masks
        if self.mask_all_delays:
            ## Mask out all delays
            X[:, batch_all_delay_masks_index] = batch_all_delay_masks

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        ## Loss over both the loss functions
        loss_recnstr = self.loss_func1(prediction, y[:, [self.sliding_window_size - 1]])
        if self.dual_loss:
            loss_entropy = self.loss_func2(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            loss = loss_recnstr + loss_entropy
        else:
            loss = loss_recnstr

        mse_loss = nn.MSELoss()
        target_size = self.sliding_window_size
        last_delay_pos = target_size - 1

        last_actual_delay = y[:, [last_delay_pos]]
        last_predicted_delay = prediction

        # Get fake prediction from mean of n-1 delays
        fake_prediction = torch.clone(y)
        fake_prediction = fake_prediction[:, :-1].mean(axis=1, keepdims=True)

        # Also use the penultimate delay as the predicted value
        penultimate_prediction = torch.clone(y)
        penultimate_prediction = penultimate_prediction[:, -2].unsqueeze(1)

        # Also use the weighted Moving average over the sequence
        ewm_data = torch.clone(y)
        ewm_data = ewm_data.cpu().numpy()

        weights = 0.99 ** np.arange(self.sliding_window_size - 1)[::-1]

        ewm_prediction = np.ma.average(ewm_data[:, :-1], axis=1, weights=weights)
        ewm_prediction = np.expand_dims(ewm_prediction, 1)

        # Also predict the media over the sequence
        median_prediction = torch.clone(y)
        median_prediction = (
            median_prediction[:, :-1].median(axis=1, keepdims=True).values
        )

        last_delay_loss = mse_loss(last_actual_delay, last_predicted_delay)

        self.log("Test loss", loss, sync_dist=True)
        return {
            "Test loss": loss,
            "last_delay_loss": last_delay_loss,
            "last_predicted_delay": last_predicted_delay,
            "last_actual_delay": last_actual_delay,
            "fake_predicted_delay": fake_prediction,
            "penultimate_predicted_delay": penultimate_prediction,
            "ewm_predicted_delay": torch.tensor(
                ewm_prediction, dtype=torch.double, device=self.device
            ),
            "median_predicted_delay": median_prediction,
        }

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
        last_delay_losses = []
        last_predicted_delay = []
        last_actual_delay = []
        fake_last_delay = []
        penultimate_predicted_delay = []
        ewm_predicted_delay = []
        median_predicted_delay = []

        for output in outputs:
            last_packet_losses = list(
                np.expand_dims(output["last_delay_loss"].cpu().detach().numpy(), 0)
            )  # Losses on last delay only
            preds = list(
                output["last_predicted_delay"].cpu().detach().numpy()
            )  # predicted last delays
            labels = list(
                output["last_actual_delay"].cpu().detach().numpy()
            )  # actual last delays
            fakes = list(
                output["fake_predicted_delay"].cpu().detach().numpy()
            )  # fake last delays
            penultimate_preds = list(
                output["penultimate_predicted_delay"].cpu().detach().numpy()
            )  # predicted penultimate delays
            ewm_preds = list(
                output["ewm_predicted_delay"].cpu().detach().numpy()
            )  # predicted ewm delays
            median_preds = list(
                output["median_predicted_delay"].cpu().detach().numpy()
            )  # predicted median delays

            last_delay_losses.extend(last_packet_losses)
            last_predicted_delay.extend(preds)
            last_actual_delay.extend(labels)
            fake_last_delay.extend(fakes)
            penultimate_predicted_delay.extend(penultimate_preds)
            ewm_predicted_delay.extend(ewm_preds)
            median_predicted_delay.extend(median_preds)

        print()
        print(
            "Check lengths for all as sanity ",
            len(last_predicted_delay),
            len(last_actual_delay),
            len(fake_last_delay),
        )

        print(
            "Mean loss on last delay (averaged from batches) is : ",
            np.mean(np.array(last_delay_losses)),
        )

        last_predicted_delay = np.array(last_predicted_delay)
        last_actual_delay = np.array(last_actual_delay)

        losses_array = np.square(np.subtract(last_predicted_delay, last_actual_delay))

        print(
            "Mean loss on last delay (averaged from items) is : ", np.mean(losses_array)
        )
        print("99%%ile loss is : ", np.quantile(losses_array, 0.99))

        fake_last_delay = np.array(fake_last_delay)
        fake_losses_array = np.square(np.subtract(fake_last_delay, last_actual_delay))

        print(
            "Mean loss on ARMA predicted last delay (averaged from items) is : ",
            np.mean(fake_losses_array),
        )
        print(
            "99%%ile loss on ARMA predicted delay is : ",
            np.quantile(fake_losses_array, 0.99),
        )

        penultimate_predicted_delay = np.array(penultimate_predicted_delay)
        penultimate_losses_array = np.square(
            np.subtract(penultimate_predicted_delay, last_actual_delay)
        )

        print(
            "Mean loss on penultimate predicted last delay (averaged from items) is : ",
            np.mean(penultimate_losses_array),
        )
        print(
            "99%%ile loss on penultimate predicted delay is : ",
            np.quantile(penultimate_losses_array, 0.99),
        )

        ewm_predicted_delay = np.array(ewm_predicted_delay)
        ewm_losses_array = np.square(
            np.subtract(ewm_predicted_delay, last_actual_delay)
        )

        print(
            "Mean loss on EWM predicted last delay (averaged from items) is : ",
            np.mean(ewm_losses_array),
        )
        print(
            "99%%ile loss on EWM predicted delay is : ",
            np.quantile(ewm_losses_array, 0.99),
        )

        median_predicted_delay = np.array(median_predicted_delay)
        median_losses_array = np.square(
            np.subtract(median_predicted_delay, last_actual_delay)
        )

        print(
            "Mean loss on median predicted as last delay (averaged from items) is : ",
            np.mean(median_losses_array),
        )
        print(
            "99%%ile loss on median predicted as delay is : ",
            np.quantile(median_losses_array, 0.99),
        )

        if self.dual_loss:
            os.system("mkdir -p plot_values_dualloss/")
            save_path = "plot_values_dualloss/"
        else:
            os.system("mkdir -p plot_values/")
            save_path = "plot_values/"
        np.save(
            save_path
            + "transformer_last_delay_window_size_{}.npy".format(
                self.sliding_window_size
            ),
            np.array(last_predicted_delay),
        )
        np.save(
            save_path
            + "arma_last_delay_window_size_{}.npy".format(self.sliding_window_size),
            np.array(fake_last_delay),
        )
        np.save(
            save_path
            + "penultimate_last_delay_window_size_{}.npy".format(
                self.sliding_window_size
            ),
            np.array(penultimate_predicted_delay),
        )
        np.save(
            save_path + "ewm_delay_window_size_{}.npy".format(self.sliding_window_size),
            np.array(ewm_predicted_delay),
        )
        np.save(
            save_path
            + "actual_last_delay_window_size_{}.npy".format(self.sliding_window_size),
            np.array(last_actual_delay),
        )
        np.save(
            save_path
            + "median_last_delay_window_size_{}.npy".format(self.sliding_window_size),
            np.array(median_predicted_delay),
        )


# Argument parser for the model and return
def get_args():
    # Hyper parameters from config file
    with open("configs/config-encoder-test.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="Transformer Encoder for delay prediction"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=float(config["weight_decay"]),
        help="Weight decay for the Adam optimizer",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(config["learning_rate"]),
        help="Learning rate for the Adam optimizer",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=float(config["dropout"]),
        help="Dropout for the transformer encoder",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=int(config["num_heads"]),
        help="Number of heads for the transformer encoder",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=int(config["num_layers"]),
        help="Number of layers for the transformer encoder",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config["epochs"]),
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config["batch_size"]),
        help="Batch size for training",
    )

    parser.add_argument(
        "--linear_size",
        type=int,
        default=int(config["linear_size"]),
        help="Linear size for the transformer encoder",
    )

    parser.add_argument(
        "--loss_function1",
        default=nn.MSELoss(),
        help="Loss function for the transformer encoder",
    )

    parser.add_argument(
        "--loss_function2",
        default=nn.CrossEntropyLoss(),
        help="Loss function for the transformer encoder",
    )

    if "loss_function" in config.keys():
        if config["loss_function"] == "huber":
            parser.add_argument(
                "--loss_function_huber",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer encoder",
            )
        if config["loss_function"] == "smoothl1":
            parser.add_argument(
                "--loss_function_smoothl1",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer encoder",
            )
        if config["loss_function"] == "kldiv":
            parser.add_argument(
                "--loss_function_kldiv",
                default=nn.KLDivLoss(),
                help="Loss function for the transformer encoder",
            )

    parser.add_argument(
        "--sliding_window_start",
        type=int,
        default=0,
        help="Start postition of the sliding window",
    )

    parser.add_argument(
        "--sliding_window_step",
        type=int,
        default=1,
        help="Shift length of the sliding window",
    )

    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=1024,
        help="Size of the sliding window",
    )

    parser.add_argument(
        "--window_batch_size",
        type=int,
        default=5000,
        help="Batch size for constructing the sliding window",
    )

    parser.add_argument(
        "--packets_per_embedding",
        type=int,
        default=21,
        help="Number of packets per embedding if fixed size aggregation",
    )

    parser.add_argument(
        "--num_bottlenecks",
        type=int,
        default=1,
        help="Number of bottlenecks in the topology",
    )

    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train the model",
    )

    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="Save the model",
    )

    parser.add_argument(
        "--make_epoch_plot",
        type=bool,
        default=False,
        help="Make the loss plot per epoch",
    )

    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        help="Test the model on test fraction of data",
    )

    parser.add_argument(
        "--test_only_new",
        type=bool,
        default=False,
        help="Test only on new data",
    )

    parser.add_argument(
        "--use_dual_loss",
        type=bool,
        default=False,
        help="Train with reconstruction loss and enropy minimization",
    )

    parser.add_argument(
        "--mask_all_sizes",
        type=bool,
        default=False,
        help="Mask out all packet sizes",
    )

    parser.add_argument(
        "--mask_all_delays",
        type=bool,
        default=False,
        help="Mask out all packet delays",
    )

    parser.add_argument(
        "--use_hierarchical_aggregation",
        type=bool,
        default=True,
        help="Use hierarchical aggregation",
    )

    parser.add_argument(
        "--pool",
        type=bool,
        default=False,
        help="Use mean pooling",
    )

    args = parser.parse_args()

    return args


# Setup and run the transformer encoder model
def main():
    # Arguments
    args = get_args()

    WEIGHTDECAY = args.weight_decay
    LEARNINGRATE = args.learning_rate
    DROPOUT = args.dropout
    NHEAD = args.num_heads
    LAYERS = args.num_layers
    EPOCHS = args.epochs
    BATCHSIZE = args.batch_size
    LINEARSIZE = args.linear_size
    LOSSFUNCTION_1 = args.loss_function1
    LOSSFUNCTION_2 = args.loss_function2

    # Params for the sliding window on the packet data
    SLIDING_WINDOW_START = args.sliding_window_start
    SLIDING_WINDOW_STEP = args.sliding_window_step
    SLIDING_WINDOW_SIZE = args.sliding_window_size
    WINDOW_BATCH_SIZE = args.window_batch_size
    PACKETS_PER_EMBEDDING = args.packets_per_embedding
    NUM_BOTTLENECKS = args.num_bottlenecks

    TRAIN = args.train
    SAVE_MODEL = args.save_model
    MAKE_EPOCH_PLOT = args.make_epoch_plot
    TEST = args.test
    TEST_ONLY_NEW = args.test_only_new

    ## Dual loss (reconstruction + entropy)
    DUAL_LOSS = args.use_dual_loss

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    if NUM_BOTTLENECKS == 1 or NUM_BOTTLENECKS == 2:
        num_features = 3  # If only timestamp, packet size and delay, else 16
    elif NUM_BOTTLENECKS == 4:
        num_features = 4  # Using receiver IP identifier (and full packet 16+3)
    input_size = SLIDING_WINDOW_SIZE * num_features
    output_size = 1

    full_feature_arr = []
    full_target_arr = []

    # If data exists in ../results in .pkl file, load it
    if not os.path.exists("../results/"):
        print("Loading data from ../results/")
        full_feature_arr = pickle.load(open("../results/full_feature_arr.pkl", "rb"))
        full_target_arr = pickle.load(open("../results/full_target_arr.pkl", "rb"))
        mean_delay = pickle.load(open("../results/mean_iat.pkl", "rb"))
        std_delay = pickle.load(open("../results/std_iat.pkl", "rb"))
    else:
        ## Get the data
        (
            full_feature_arr,
            full_target_arr,
            mean_delay,
            std_delay,
            all_values,
        ) = generate_sliding_windows(
            args.sliding_window_size,
            args.window_batch_size,
            num_features,
            args.test_only_new,
            args.num_bottlenecks,
            reduce_type=True,
            MEMENTO=True,
            IAT_LABEL=False,
            DATACENTER_BURSTS=False,
            LAPTOP_ON_WIFI=False,
            RTT_LABEL=False,
            RTT_WIFI_NETWORK=False,
        )

    ## Model definition with delay scaling params
    model = TransformerEncoder(
        input_size=input_size,
        target_size=output_size,
        loss_function1=LOSSFUNCTION_1,
        loss_function2=LOSSFUNCTION_2,
        delay_mean=mean_delay,
        delay_std=std_delay,
        packets_per_embedding=PACKETS_PER_EMBEDDING,
        layers=LAYERS,
        num_heads=NHEAD,
        dropout=DROPOUT,
        weight_decay=WEIGHTDECAY,
        learning_rate=LEARNINGRATE,
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        linear_size=LINEARSIZE,
        sliding_window_size=SLIDING_WINDOW_SIZE,
        dual_loss=DUAL_LOSS,
        mask_all_sizes=args.mask_all_sizes,
        mask_all_delays=args.mask_all_delays,
        use_hierarchical_aggregation=args.use_hierarchical_aggregation,
        pool=args.pool,
    )

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

    """## Take 10% fine-tuning data only
    train_vectors = train_vectors[:int(0.1*len(train_vectors))]
    train_labels = train_labels[:int(0.1*len(train_labels))]"""

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    test_dataset = PacketDataset(test_vectors, test_labels)
    # print(train_dataset.__getitem__(0))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )

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

    # Save path as per single or dual loss
    # Make new dir for storing logs
    if DUAL_LOSS:
        os.system("mkdir -p encoder_delay_logs_dualloss/")
        save_path = "encoder_delay_logs_dualloss/"
    else:
        os.system("mkdir -p encoder_delay_logs/")
        save_path = "encoder_delay_logs/"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_path)

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=16,
            accelerator="gpu",
            devices=[1],
            # strategy="ddp",
            max_epochs=EPOCHS,
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
        # Load checkpoint if exists
        if os.path.exists(
            "{}/iat_pred_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
        ):
            print("Loading checkpoint")
            cpath = "{}/iat_pred_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
            model = TransformerEncoder.load_from_checkpoint(
                input_size=input_size,
                target_size=output_size,
                loss_function1=LOSSFUNCTION_1,
                loss_function2=LOSSFUNCTION_2,
                delay_mean=mean_delay,
                delay_std=std_delay,
                packets_per_embedding=PACKETS_PER_EMBEDDING,
                layers=LAYERS,
                num_heads=NHEAD,
                dropout=DROPOUT,
                weight_decay=WEIGHTDECAY,
                learning_rate=LEARNINGRATE,
                epochs=EPOCHS,
                batch_size=BATCHSIZE,
                linear_size=LINEARSIZE,
                sliding_window_size=SLIDING_WINDOW_SIZE,
                dual_loss=DUAL_LOSS,
                mask_all_sizes=False,
                mask_all_delays=False,
                use_hierarchical_aggregation=True,
                pool=False,
                checkpoint_path=cpath,
                strict=True,
            )
            model.train()
        else:
            print("No checkpoint found, training from scratch")
            model.train()
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf {}/lightning_logs/".format(save_path))

        trainer.fit(model, train_loader, val_loader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint(
            "{}/iat_pred_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
        )

    if SAVE_MODEL:
        pass
        # torch.save(model, "encoder_delay_logs/finetuned_encoder_scratch.pt")

    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "{}/lightning_logs/version_0".format(save_path)
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
            print("Loading checkpoint")
            cpath = "{}iat_pred_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
            testmodel = TransformerEncoder.load_from_checkpoint(
                input_size=input_size,
                target_size=output_size,
                loss_function1=LOSSFUNCTION_1,
                loss_function2=LOSSFUNCTION_2,
                delay_mean=mean_delay,
                delay_std=std_delay,
                packets_per_embedding=PACKETS_PER_EMBEDDING,
                layers=LAYERS,
                num_heads=NHEAD,
                dropout=DROPOUT,
                weight_decay=WEIGHTDECAY,
                learning_rate=LEARNINGRATE,
                epochs=EPOCHS,
                batch_size=BATCHSIZE,
                linear_size=LINEARSIZE,
                sliding_window_size=SLIDING_WINDOW_SIZE,
                dual_loss=DUAL_LOSS,
                mask_all_sizes=False,
                mask_all_delays=False,
                use_hierarchical_aggregation=True,
                pool=False,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            if TEST_ONLY_NEW:
                new_test_dataset = PacketDataset(full_feature_arr, full_target_arr)
                new_test_loader = DataLoader(
                    new_test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
                )

                trainer.test(testmodel, dataloaders=new_test_loader)

            else:
                trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()
