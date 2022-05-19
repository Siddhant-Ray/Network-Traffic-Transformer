import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

frame = pd.read_csv("topo_test_1.csv")
# Get the time stamp, packet size and delay (from my format, Alex uses a different format)
frame = frame[frame.columns[[1,7,-6]]]
frame.columns = ["t", "delay", "size"]
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

sbs.fig.suptitle('Bottleneck queue plot with multiple senders')
plt.savefig("Queuesize"+".png")


dropframe = pd.read_csv("drops.csv", names=["source", "time", "packetsize"])

print("Drop fraction:", len(dropframe) / (len(dropframe) + len(frame)))
