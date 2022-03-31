import os
import pandas as pd

print("Current directory is:", os.getcwd())
print("Generate combined csv for TCP congestion data")

path = "/local/home/sidray/packet_transformer/outputs/congestion_2/"

router_drop_df = pd.read_csv(path+"RxDrops_lrouter_0.csv")
print(router_drop_df.head())

router_revd_df = pd.read_csv(path+"RxRevd_lrouter_0.csv")
print(router_revd_df.head())

router_sent_df = pd.read_csv(path+"TxSent_lrouter_0.csv")
print(router_sent_df.head())

numsenders = 1
currentsender = 0

while (currentsender < numsenders):
    sender_id = str(currentsender)
    sender_sent_df = pd.read_csv(path+"TxSent_sender_" + sender_id + ".csv")
    print(sender_sent_df.head())
    currentsender+=1

print("preliminary test done...")