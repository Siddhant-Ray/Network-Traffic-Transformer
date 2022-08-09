# Orignal author: Siddhant Ray

from csv import reader
from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


mct_log_dir_pretrained0 = "../../logs/finetune_mct_logs/"
reader_pretrained0 = SummaryReader(mct_log_dir_pretrained0)
df_pretrained0 = reader_pretrained0.scalars


train_loss_epoch_df_pretrained0= df_pretrained0[df_pretrained0["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained0.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained0 = df_pretrained0[df_pretrained0["tag"] == "Val loss"]
val_loss_epoch_df_pretrained0.reset_index(inplace=True, drop=True)


mct_log_dir_pretrained1 = "../../logs/finetune_mct_logs3/"
reader_pretrained1 = SummaryReader(mct_log_dir_pretrained1)
df_pretrained1 = reader_pretrained1.scalars

train_loss_epoch_df_pretrained1= df_pretrained1[df_pretrained1["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained1.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained1 = df_pretrained1[df_pretrained1["tag"] == "Val loss"]
val_loss_epoch_df_pretrained1.reset_index(inplace=True, drop=True)

mct_log_dir_pretrained2 = "../../logs/finetune_mct_logs4/"
reader_pretrained2 = SummaryReader(mct_log_dir_pretrained2)
df_pretrained2 = reader_pretrained2.scalars

train_loss_epoch_df_pretrained2 = df_pretrained2[df_pretrained2["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained2.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained2 = df_pretrained2[df_pretrained2["tag"] == "Val loss"]
val_loss_epoch_df_pretrained2.reset_index(inplace=True, drop=True)

# Print df shape 
print(train_loss_epoch_df_pretrained2.shape)
print(val_loss_epoch_df_pretrained2.shape)

sns.set_theme("paper", "whitegrid", font_scale=1.5)
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
    'savefig.pad_inches': 0.01,
})

# Test figure, makse subplots for train and val 
fig, ax = plt.subplots(2,figsize=(5, 5), sharex=True)
t0 = sns.lineplot(x=train_loss_epoch_df_pretrained0.index, y="value", data=train_loss_epoch_df_pretrained0, color = 'blue', label="Fixed Mask Last", ax = ax[0])
t1 = sns.lineplot(x=train_loss_epoch_df_pretrained1.index, y="value", data=train_loss_epoch_df_pretrained1, color = 'green', label="Last 16", ax = ax[0])
t2 = sns.lineplot(x=train_loss_epoch_df_pretrained2.index, y="value", data=train_loss_epoch_df_pretrained2, color = 'red', label="Last 32", ax = ax[0])
# Label plot 
ax[0].set_xlabel("Training Epoch", fontsize=10)
ax[0].set_ylabel("Training MSE",fontsize=10)
ax[0].lines[0].set_linestyle("dotted")
ax[0].lines[1].set_linestyle("--")
ax[0].lines[2].set_linestyle("-.")
ticks = [0, 0.25, 0.5, 0.75, 1]
ax[0].yaxis.set_ticks(ticks)
tickLabels = map(str, ticks)
ax[0].yaxis.set_ticklabels(tickLabels)
ax[0].axis(ymin=0,ymax=1)
ax[0].axis(xmin=0,xmax=18)

v0 = sns.lineplot(x=val_loss_epoch_df_pretrained0.index, y="value", data=val_loss_epoch_df_pretrained0, color = 'blue', label="Fixed Mask Last", ax = ax[1])
v1 = sns.lineplot(x=val_loss_epoch_df_pretrained1.index, y="value", data=val_loss_epoch_df_pretrained1, color = 'green', label="Last 16", ax = ax[1])
v2 = sns.lineplot(x=val_loss_epoch_df_pretrained2.index, y="value", data=val_loss_epoch_df_pretrained2, color = 'red', label="Last 32", ax = ax[1])

# Label plot
ax[1].set_xlabel("Training Epoch", fontsize=10)
ax[1].set_ylabel("Validation MSE",fontsize=10)
ax[1].lines[0].set_linestyle("dotted")
ax[1].lines[1].set_linestyle("--")
ax[1].lines[2].set_linestyle("-.")
ticks = [0, 0.25, 0.5, 0.75, 1]
ax[1].yaxis.set_ticks(ticks)
tickLabels = map(str, ticks)
ax[1].yaxis.set_ticklabels(tickLabels)
ax[1].axis(ymin=0,ymax=1)
ax[1].axis(xmin=0,xmax=18)

fig.legend(["Fixed Mask Last", "Last 16", "Last 32"],loc = "upper right", bbox_to_anchor=(0.968, 0.958), ncol=1, fontsize=10)
ax[1].get_legend().remove()
ax[0].get_legend().remove()
fig.tight_layout()
fig.savefig("../../figures_test/finetune_mct_loss_comparison.pdf")

## Create the same for aggregated masking also 
# Create a new dataframe for aggregated masking

mct_log_dir_pretrained3 = "../../logs/finetune_mct_logs5/"
reader_pretrained3 = SummaryReader(mct_log_dir_pretrained3)
df_pretrained3 = reader_pretrained3.scalars

train_loss_epoch_df_pretrained3 = df_pretrained3[df_pretrained3["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained3.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained3 = df_pretrained3[df_pretrained3["tag"] == "Val loss"]
val_loss_epoch_df_pretrained3.reset_index(inplace=True, drop=True)

mct_log_dir_pretrained4 = "../../logs/finetune_mct_logs6/"
reader_pretrained4 = SummaryReader(mct_log_dir_pretrained4)
df_pretrained4 = reader_pretrained4.scalars

train_loss_epoch_df_pretrained4 = df_pretrained4[df_pretrained4["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained4.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained4 = df_pretrained4[df_pretrained4["tag"] == "Val loss"]
val_loss_epoch_df_pretrained4.reset_index(inplace=True, drop=True)

mct_log_dir_pretrained5 = "../../logs/finetune_mct_logs7/"
reader_pretrained5 = SummaryReader(mct_log_dir_pretrained5)
df_pretrained5 = reader_pretrained5.scalars

train_loss_epoch_df_pretrained5 = df_pretrained5[df_pretrained5["tag"] == "Avg loss per epoch"]
train_loss_epoch_df_pretrained5.reset_index(inplace=True, drop=True)

val_loss_epoch_df_pretrained5 = df_pretrained5[df_pretrained5["tag"] == "Val loss"]
val_loss_epoch_df_pretrained5.reset_index(inplace=True, drop=True)


## Plot the loss for aggregated masking
fig, ax = plt.subplots(2,figsize=(5, 5), sharex=True)
t0 = sns.lineplot(x=train_loss_epoch_df_pretrained0.index, y="value", data=train_loss_epoch_df_pretrained0, color = 'blue', label="Fixed Mask Last", ax = ax[0])
t1 = sns.lineplot(x=train_loss_epoch_df_pretrained4.index, y="value", data=train_loss_epoch_df_pretrained4, color = 'green', label="Choose Enc. State", ax = ax[0])
t2 = sns.lineplot(x=train_loss_epoch_df_pretrained3.index, y="value", data=train_loss_epoch_df_pretrained3, color = 'red', label="Choose Agg. Level", ax = ax[0])

# Label plot
ax[0].set_xlabel("Training Epoch", fontsize=10)
ax[0].set_ylabel("Training MSE",fontsize=10)
ax[0].lines[0].set_linestyle("dotted")
ax[0].lines[1].set_linestyle("--")
ax[0].lines[2].set_linestyle("-.")
ticks = [0, 0.25, 0.5, 0.75, 1]
ax[0].yaxis.set_ticks(ticks)
tickLabels = map(str, ticks)
ax[0].yaxis.set_ticklabels(tickLabels)
ax[0].axis(ymin=0,ymax=1)
ax[0].axis(xmin=0,xmax=13)

v0 = sns.lineplot(x=val_loss_epoch_df_pretrained0.index, y="value", data=val_loss_epoch_df_pretrained0, color = 'blue', label="Fixed Mask Last", ax = ax[1])
v1 = sns.lineplot(x=val_loss_epoch_df_pretrained4.index, y="value", data=val_loss_epoch_df_pretrained4, color = 'green', label="Choose Enc. State", ax = ax[1])
v2 = sns.lineplot(x=val_loss_epoch_df_pretrained3.index, y="value", data=val_loss_epoch_df_pretrained3, color = 'red', label="Choose Agg. Level", ax = ax[1])

# Label plot
ax[1].set_xlabel("Training Epoch", fontsize=10)
ax[1].set_ylabel("Validation MSE",fontsize=10)
ax[1].lines[0].set_linestyle("dotted")
ax[1].lines[1].set_linestyle("--")
ax[1].lines[2].set_linestyle("-.")
ticks = [0, 0.25, 0.5, 0.75, 1]
ax[1].yaxis.set_ticks(ticks)
tickLabels = map(str, ticks)
ax[1].yaxis.set_ticklabels(tickLabels)
ax[1].axis(ymin=0,ymax=1)
ax[1].axis(xmin=0,xmax=18)

# Make legend
fig.legend(["Fixed Mask Last","Choose Enc. State","Choose Agg. Level"],loc = "upper right", bbox_to_anchor=(0.968, 0.958), ncol=1, fontsize=10)
ax[1].get_legend().remove()
ax[0].get_legend().remove()
fig.tight_layout()
fig.savefig("../../figures_test/finetune_mct_loss_comparison_agg.pdf")








