from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

log_dir = "../../logs/encoder_delay_logs/"
reader = SummaryReader(log_dir)
df = reader.scalars

epoch_train_loss_df = df[df["tag"] == "Avg loss per epoch"]
epoch_train_loss_df.reset_index(inplace=True, drop=True)
print(epoch_train_loss_df)

train_loss_step_df = df[df["tag"] == "Train loss"]
train_loss_step_df.reset_index(inplace=True, drop=True)
print(train_loss_step_df)

val_loss_step_df = df[df["tag"] == "Val loss"]
val_loss_step_df.reset_index(inplace=True, drop=True)
print(val_loss_step_df)

## Train loss plot (pre-training)
plt.figure(figsize=(5, 5))
sns.lineplot(x=epoch_train_loss_df.index, y="value", data=epoch_train_loss_df, label="Avg train loss per epoch")
plt.show()

## Val loss plot (pre-training)
plt.figure(figsize=(5, 5))
sns.lineplot(x=val_loss_step_df.index, y="value", data=val_loss_step_df, label="Avg val loss per epoch")
plt.show()

mct_log_dir_pretrained = "../../logs/finetune_mct_logs/"
reader_pretrained = SummaryReader(mct_log_dir_pretrained)
df_pretrained = reader_pretrained.scalars

train_loss_epoch_df_pretrained = df_pretrained[df_pretrained["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained = df_pretrained[df_pretrained["tag"] == "Val loss"]
val_loss_epoch_df_pretrained.reset_index(inplace=True, drop=True)

## Train loss (pre-trained)
plt.figure(figsize=(5, 5))
sns.lineplot(x=train_loss_epoch_df_pretrained.index, y="value", data=train_loss_epoch_df_pretrained, label="Avg train loss per epoch")
plt.title("Train loss on MCT prediction (pre-trained)")
plt.show()

## Val loss (pre-trained)
plt.figure(figsize=(5, 5))
sns.lineplot(x=val_loss_epoch_df_pretrained.index, y="value", data=val_loss_epoch_df_pretrained, label="Avg val loss per epoch")
plt.title("Val loss on MCT prediction (pre-trained)")
plt.show()

mct_log_dir_nonpretrained = "../../logs/finetune_mct_logs2/"
reader_nonpretrained = SummaryReader(mct_log_dir_nonpretrained)
df_nonpretrained = reader_nonpretrained.scalars

train_loss_epoch_df_nonpretrained = df_nonpretrained[df_nonpretrained["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_nonpretrained.reset_index(inplace=True, drop=True)

val_loss_epoch_df_nonpretrained = df_nonpretrained[df_nonpretrained["tag"] == "Val loss"]
val_loss_epoch_df_nonpretrained.reset_index(inplace=True, drop=True)

## Train loss (non-pretrained)
plt.figure(figsize=(5, 5))
sns.lineplot(x=train_loss_epoch_df_nonpretrained.index, y="value", data=train_loss_epoch_df_nonpretrained, label="Avg train loss per epoch")
plt.title("Train loss on MCT prediction (non-pretrained)")
plt.show()

## Val loss (non-pretrained)
plt.figure(figsize=(5, 5))
sns.lineplot(x=val_loss_epoch_df_nonpretrained.index, y="value", data=val_loss_epoch_df_nonpretrained, label="Avg val loss per epoch")
plt.title("Val loss on MCT prediction (non-pretrained)")
plt.show()




