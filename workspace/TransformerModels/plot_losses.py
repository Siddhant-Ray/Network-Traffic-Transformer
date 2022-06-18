from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# plt.show()

## Val loss plot (pre-training)
plt.figure(figsize=(5, 5))
sns.lineplot(x=val_loss_step_df.index, y="value", data=val_loss_step_df, label="Avg val loss per epoch")
# plt.show()

mct_log_dir_pretrained = "../../logs/finetune_mct_logs/"
reader_pretrained = SummaryReader(mct_log_dir_pretrained)
df_pretrained = reader_pretrained.scalars

train_loss_epoch_df_pretrained = df_pretrained[df_pretrained["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained = df_pretrained[df_pretrained["tag"] == "Val loss"]
val_loss_epoch_df_pretrained.reset_index(inplace=True, drop=True)


mct_log_dir_nonpretrained = "../../logs/finetune_mct_logs2/"
reader_nonpretrained = SummaryReader(mct_log_dir_nonpretrained)
df_nonpretrained = reader_nonpretrained.scalars

train_loss_epoch_df_nonpretrained = df_nonpretrained[df_nonpretrained["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_nonpretrained.reset_index(inplace=True, drop=True)

val_loss_epoch_df_nonpretrained = df_nonpretrained[df_nonpretrained["tag"] == "Val loss"]
val_loss_epoch_df_nonpretrained.reset_index(inplace=True, drop=True)

sns.set_theme("paper", "whitegrid")
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',

    'lines.linewidth': 2,
    'lines.markeredgewidth': 0,

    'scatter.marker': '.',
    'scatter.edgecolors': 'none',

    # Set image quality and reduce whitespace around saved figure.
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

fig, ax = plt.subplots(2,figsize=(3, 3.35))

## Train loss (pre-trained vs non-pretrained)
# plt.figure(figsize=(3, 1.67))
g1 = sns.lineplot(x=train_loss_epoch_df_pretrained.index, y="value", data=train_loss_epoch_df_pretrained, label="Pre-trained", ax=ax[0])
g2 = sns.lineplot(x=train_loss_epoch_df_nonpretrained.index, y="value", data=train_loss_epoch_df_nonpretrained, label="From scratch",ax=ax[0] )
# plt.title("Train loss on MCT prediction pre-trained vs non-pretrained")
ax[0].set_xlabel("Training Epoch", fontsize=8)
ax[0].set_ylabel("Training MSE",fontsize=8)
ax[0].axis(ymin=0,ymax=1)
ax[0].axis(xmin=0,xmax=25)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# ax[0].legend(fontsize=8)
# plt.savefig("../../figures/MCT_train_loss.pdf")

## Val loss (pre-trained vs non-pretrained)
# plt.figure(figsize=(3, 1.67))
g3 = sns.lineplot(x=val_loss_epoch_df_pretrained.index, y="value", data=val_loss_epoch_df_pretrained, label="Pre-trained", ax=ax[1])
g4 = sns.lineplot(x=val_loss_epoch_df_nonpretrained.index, y="value", data=val_loss_epoch_df_nonpretrained, label="From scratch", ax=ax[1])
# plt.title("Val loss on MCT prediction pre-trained vs non-pretrained")
ax[1].set_xlabel("Training Epoch",fontsize=8)
ax[1].set_ylabel("Validation MSE", fontsize=8)
ax[1].axis(ymin=0,ymax=1)
ax[1].axis(xmin=0,xmax=25)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# ax[1].legend(fontsize=8)
fig.legend(["Pre-trained", "From scratch"],loc = "upper right", bbox_to_anchor=(0.94, 0.955), ncol=1, fontsize=6)
ax[1].get_legend().remove()
ax[0].get_legend().remove()
fig.tight_layout()
plt.savefig("../../figures/MCT_loss.pdf")




