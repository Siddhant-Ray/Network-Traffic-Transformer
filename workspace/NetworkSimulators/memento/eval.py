import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

BIG = True
TEST = True # Marked true for fine-tuning data with multiple bottlenecks
val = sys.argv[1]

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

if not TEST:
    frame = pd.read_csv("small_test_no_disturbance_with_message_ids{}.csv".format(val))
else:
    if not BIG:
        frame = pd.read_csv("small_test_one_disturbance_with_message_ids{}.csv".format(val))
    else:
        frame = pd.read_csv("large_test_disturbance_with_message_ids{}.csv".format(val))

# Get the time stamp, packet size and delay (from my format, Alex uses a different format)
frame = frame[frame.columns[[1,7,-8]]]
frame.columns = ["t", "size", "delay"]
print(frame.head())

frame = (
    frame
    .assign(delay=lambda df: df['delay'])  # to ms.
)

plt.figure(figsize=(5,5))
sbs = sns.displot(
    data=frame,
    kind='ecdf',
    x='delay'
)

sbs.fig.suptitle('Delay plot with multiple senders')
sbs.set(xlabel='Delay (seconds)', ylabel='Fraction of packets')
plt.xlim([0,0.5])
plt.ylim(bottom=0)
# Tight layout
sbs.fig.tight_layout()
plt.savefig("delay"+".pdf")


frame['delay'].quantile([0.5, 0.99])

throughput = frame.loc[frame['t'] > 20, 'size'].sum() / 40 / (1024*1024)  # in MBps

queueframe = pd.read_csv("queue.csv", names=["source", "time", "size"])

bottleneck_source = "/NodeList/0/DeviceList/0/$ns3::CsmaNetDevice/TxQueue/PacketsInQueue"
bottleneck_queue = queueframe[queueframe["source"] == bottleneck_source]
print(bottleneck_source)

plt.figure(figsize=(5,5))
scs = sns.relplot(
    data=bottleneck_queue,
    kind='line',
    x='time',
    y='size',
    legend=False,
    ci=None,
)

scs.fig.suptitle('Bottleneck queue plot with multiple senders')
plt.savefig("Queuesize"+".pdf")

## Bottleneck plots for switches A, B, D, G

if BIG:
    values = [6, 7, 9, 12]
    dict_switches = {
        6: "A",
        7: "B",
        9: "D",
        12: "G"
    }
else:
    values = [2, 3]
    dict_switches = {
        2: "A",
        3: "B"
    }

for value in values:
    bottleneck_source = "/NodeList/{}/DeviceList/0/$ns3::CsmaNetDevice/TxQueue/PacketsInQueue".format(value)
    bottleneck_queue = queueframe[queueframe["source"] == bottleneck_source]
    print(bottleneck_source)

    plt.figure(figsize=(5,5))
    scs = sns.relplot(
        data=bottleneck_queue,
        kind='line',
        x='time',
        y='size',
        legend=False,
        ci=None,
    )

    scs.fig.suptitle('Bottleneck queue on switch {} '.format(dict_switches[value]))
    scs.fig.suptitle('Queue on bottleneck switch')
    scs.set(xlabel='Simulation Time (seconds)', ylabel='Queue Size (packets)')
    plt.xlim([0,60])
    plt.ylim([0,1000])
    
    save_name = "Queue profile on switch {}".format(dict_switches[value]) + ".pdf"
    scs.fig.tight_layout()
    plt.savefig(save_name) 

dropframe = pd.read_csv("drops.csv", names=["source", "time", "packetsize"])

print("Drop fraction:", len(dropframe) / (len(dropframe) + len(frame)))

## Plot delay distribution for each receiver
new_frame = pd.read_csv("large_test_disturbance_with_message_ids{}.csv".format(val))
new_frame = new_frame[new_frame.columns[[1,7, 23, -8]]]
new_frame.columns = ["t", "size", "dest ip", "delay"]
print(new_frame.head())

gb = new_frame.groupby('dest ip')    
groups = [gb.get_group(x) for x in gb.groups]
print(groups)

for idx, group in enumerate(groups):
    print(idx, group.shape)
    plt.figure(figsize=(5,5))
    scs = sns.displot(
            data=group,
            kind='ecdf',
            x='delay',
            legend=False
        )

    scs.fig.suptitle('Delay plot on receiver {} '.format(idx+1))
    scs.set(xlabel='Delay', ylabel='Fraction of packets')
    plt.xlim([0,0.5])
    # Tight layout
    scs.fig.tight_layout()
    plt.savefig("delay_Receiver{}".format(idx)+".pdf")

        
