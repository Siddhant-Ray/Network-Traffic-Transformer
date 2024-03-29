# Orignal author: Siddhant Ray

import argparse
import copy
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
from generate_sequences import generate_sliding_windows
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
from utils import (PacketDataset, PacketDatasetEncoder,
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
LOSSFUNCTION = nn.MSELoss()

if "loss_function" in config.keys():
    if config["loss_function"] == "huber":
        LOSSFUNCTION = nn.HuberLoss()
    if config["loss_function"] == "smoothl1":
        LOSSFUNCTION = nn.SmoothL1Loss()
    if config["loss_function"] == "kldiv":
        LOSSFUNCTION = nn.KLDivLoss()

# Params for the sliding window on the packet data
SLIDING_WINDOW_START = 0
SLIDING_WINDOW_STEP = 1
SLIDING_WINDOW_SIZE = 1024
WINDOW_BATCH_SIZE = 5000
PACKETS_PER_EMBEDDING = 25
NUM_BOTTLENECKS = 4

TRAIN = True
PRETRAINED = True
SAVE_MODEL = True
MAKE_EPOCH_PLOT = False
TEST = True
TEST_ONLY_NEW = False

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

        # Predict the output

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
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=True, device=self.device
        )
        batch_mask = batch_mask.double()
        X[:, [batch_mask_index]] = batch_mask

        # Every packet separately into the transformer (project to linear if needed)
        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        loss = self.loss_func(prediction, y[:, [SLIDING_WINDOW_SIZE - 1]])
        self.log("Train loss", loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        X, y = val_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=False, device=self.device
        )
        batch_mask = batch_mask.double()
        X[:, [batch_mask_index]] = batch_mask

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        loss = self.loss_func(prediction, y[:, [SLIDING_WINDOW_SIZE - 1]])
        self.log("Val loss", loss, sync_dist=True)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y = test_batch

        batch_mask_index = self.input_size - 1
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=False, device=self.device
        )
        batch_mask = batch_mask.double()
        X[:, [batch_mask_index]] = batch_mask

        prediction = self.forward(X)

        ## Un-normalize the delay prediction
        prediction = prediction * self.delay_std + self.delay_mean

        loss = self.loss_func(prediction, y[:, [SLIDING_WINDOW_SIZE - 1]])

        mse_loss = nn.MSELoss()
        target_size = SLIDING_WINDOW_SIZE
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

        weights = 0.99 ** np.arange(SLIDING_WINDOW_SIZE - 1)[::-1]

        ewm_prediction = np.ma.average(ewm_data[:, :-1], axis=1, weights=weights)
        ewm_prediction = np.expand_dims(ewm_prediction, 1)

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

        for output in outputs:
            last_packet_losses = list(
                output["last_delay_loss"].cpu().detach().numpy()
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
            )  # predicted penultimate delays

            last_delay_losses.extend(last_packet_losses)
            last_predicted_delay.extend(preds)
            last_actual_delay.extend(labels)
            fake_last_delay.extend(fakes)
            penultimate_predicted_delay.extend(penultimate_preds)
            ewm_predicted_delay.extend(ewm_preds)

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

        save_path = "plot_values_finetune/3features/"
        np.save(
            save_path
            + "transformer_last_delay_window_size_{}.npy".format(SLIDING_WINDOW_SIZE),
            np.array(last_predicted_delay),
        )
        np.save(
            save_path
            + "arma_last_delay_window_size_{}.npy".format(SLIDING_WINDOW_SIZE),
            np.array(fake_last_delay),
        )
        np.save(
            save_path
            + "penultimate_last_delay_window_size_{}.npy".format(SLIDING_WINDOW_SIZE),
            np.array(penultimate_predicted_delay),
        )
        np.save(
            save_path + "ewm_delay_window_size_{}.npy".format(SLIDING_WINDOW_SIZE),
            np.array(ewm_predicted_delay),
        )
        np.save(
            save_path
            + "actual_last_delay_window_size_{}.npy".format(SLIDING_WINDOW_SIZE),
            np.array(last_actual_delay),
        )


def main():

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    # Hack this a bit due to size mismatch, should be fixed in the future!!
    if NUM_BOTTLENECKS == 1 or NUM_BOTTLENECKS == 2:
        num_features = 3  # If only timestamp, packet size and delay, else 16
        input_size = sl_win_size * num_features
    elif NUM_BOTTLENECKS == 4:
        # Using receiver IP identifier, keep size as same for loading checkpoint, overwrite custom layers
        # Should be 4 here, but we keep it 3 to enable checkpoint loading and not run into size mismatch
        # We overwrite custom layers in the model class later, as they are only the linear layers, not Transformer layers
        num_features = 4
        input_size = sl_win_size * (num_features - 1)  # Keep this as 3
    output_size = 1

    full_feature_arr = []
    full_target_arr = []

    ## Get the data
    full_feature_arr, full_target_arr, mean_delay, std_delay = generate_sliding_windows(
        SLIDING_WINDOW_SIZE,
        WINDOW_BATCH_SIZE,
        num_features,
        TEST_ONLY_NEW,
        NUM_BOTTLENECKS,
    )
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
            pool=False,
            checkpoint_path=cpath,
            strict=True,
        )

        # Freeze or unfreeze everything!!
        for params in model.parameters():
            params.requires_grad = True

        # Unfreeze the linear layers
        for params in model.norm1.parameters():
            params.requires_grad = True
        for params in model.norm2.parameters():
            params.requires_grad = True
        for params in model.encoderpred1.parameters():
            params.requires_grad = True
        for params in model.encoderpred2.parameters():
            params.requires_grad = True
        for params in model.linear1.parameters():
            params.requires_grad = True
        for params in model.linear2.parameters():
            params.requires_grad = True
        for params in model.activ1.parameters():
            params.requires_grad = True
        for params in model.activ2.parameters():
            params.requires_grad = True
        for params in model.activ3.parameters():
            params.requires_grad = True

    else:
        # Do not freeeze anything for non pre-trained
        model = TransformerEncoder(
            input_size,
            output_size,
            LOSSFUNCTION,
            mean_delay,
            std_delay,
            PACKETS_PER_EMBEDDING,
            pool=False,
        )

    ## Re-write model to use custom layers with new sizes (if bigger toopology only)
    if NUM_BOTTLENECKS == 4:
        new_packet_size = 4
        new_input_size = sl_win_size * new_packet_size
        model.feature_transform1 = nn.Sequential(
            Rearrange(
                "b (seq feat) -> b seq feat",
                seq=SLIDING_WINDOW_SIZE,
                feat=new_packet_size,
            ),  # Make 1000
            nn.Linear(new_packet_size, LINEARSIZE),
            nn.LayerNorm(LINEARSIZE),  # pre-normalization
        )
        model.encoderpred1 = nn.Linear(LINEARSIZE, new_input_size // 8)
        model.encoderpred2 = nn.Linear(new_input_size // 8, output_size)

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

    # Make new dir for storing logs
    os.system("mkdir -p finetune_encoder_logs/")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="finetune_encoder_logs/")

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=16,
            gpus=-1,
            strategy="dp",
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
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf finetune_encoder_logs/lightning_logs/")

        trainer.fit(model, train_loader, val_loader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint(
            "finetune_encoder_logs/finetune_nonpretrained_window{}.ckpt".format(
                SLIDING_WINDOW_SIZE
            )
        )

    if SAVE_MODEL:
        pass
        # torch.save(model, "encoder_delay_logs/finetuned_encoder_scratch.pt")

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
        if TRAIN:
            trainer.test(model, dataloaders=test_loader)
        else:

            model.eval()

            if TEST_ONLY_NEW:

                cpath = (
                    "finetune_encoder_logs/finetune_nonpretrained_window{}.ckpt".format(
                        SLIDING_WINDOW_SIZE
                    )
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

                new_test_dataset = PacketDataset(full_feature_arr, full_target_arr)
                new_test_loader = DataLoader(
                    new_test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
                )

                trainer.test(testmodel, dataloaders=new_test_loader)

            else:
                trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
