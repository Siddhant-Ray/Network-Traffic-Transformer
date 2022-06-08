import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import sys

BIG = False
TEST = True
val = sys.argv[1]

if not TEST:
    frame = pd.read_csv("small_test_no_disturbance{}.csv".format(val))
else:
    frame = pd.read_csv("small_test_one_disturbance{}.csv".format(val))
# Get the time stamp, packet size and delay (from my format, Alex uses a different format)
frame = frame[frame.columns[[1,7,-8]]]
frame.columns = ["t", "size", "delay"]
print(frame.head())

frame = (
    frame
    .assign(delay=lambda df: df['delay'] * 1000)  # to ms.
)

plt.figure()
sbs = sns.displot(
    data=frame,
    kind='ecdf',
    x='delay'
)

sbs.fig.suptitle('Delay plot with multiple senders')

plt.savefig("delay"+".png")


frame['delay'].quantile([0.5, 0.99])

throughput = frame.loc[frame['t'] > 20, 'size'].sum() / 40 / (1024*1024)  # in MBps

queueframe = pd.read_csv("queue.csv", names=["source", "time", "size"])

bottleneck_source = "/NodeList/0/DeviceList/0/$ns3::CsmaNetDevice/TxQueue/PacketsInQueue"
bottleneck_queue = queueframe[queueframe["source"] == bottleneck_source]
print(bottleneck_source)

plt.figure()
scs = sns.relplot(
    data=bottleneck_queue,
    kind='line',
    x='time',
    y='size',
    legend=False,
    ci=None,
)

scs.fig.suptitle('Bottleneck queue plot with multiple senders')
plt.savefig("Queuesize"+".png")

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

    plt.figure()
    scs = sns.relplot(
        data=bottleneck_queue,
        kind='line',
        x='time',
        y='size',
        legend=False,
        ci=None,
    )

    scs.fig.suptitle('Bottleneck queue on switch {} '.format(dict_switches[value]))
    save_name = "Queue profile on switch {}".format(dict_switches[value]) + ".png"
    plt.savefig(save_name) 

dropframe = pd.read_csv("drops.csv", names=["source", "time", "packetsize"])

print("Drop fraction:", len(dropframe) / (len(dropframe) + len(frame)))
