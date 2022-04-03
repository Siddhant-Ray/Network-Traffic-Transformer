import os
import pandas as pd

print("Current directory is:", os.getcwd())
print("Generate combined csv for TCP congestion data")

path = "/local/home/sidray/packet_transformer/outputs/congestion_2/"

df_cols = ["Timestamp", "Queue Size", "Packets dropped", "Flow ID", "Packet ID", "Packet Size"]
cols_to_drop = [0,2,4,6,8,10]

router_drop_df = pd.read_csv(path+"RxDrops_lrouter_0.csv")
router_drop_descol = router_drop_df.iloc[1][0].split(" ")
router_drop_descol_value = ( " ".join(router_drop_descol[0:2]))
router_drop_df.drop(router_drop_df.columns[cols_to_drop],axis=1,inplace=True)
router_drop_df.columns = df_cols
router_drop_df["Decision"] = router_drop_descol_value
print(router_drop_df.head())
print(router_drop_df.shape)

df_cols = ["Timestamp", "Queue Size", "Flow ID", "Packet ID", "Packet Size"]
cols_to_drop = [0,2,4,6,8]
router_revd_df = pd.read_csv(path+"RxRevd_lrouter_0.csv")
router_revd_descol = router_revd_df.iloc[1][0].split(" ")
router_revd_descol_value = ( " ".join(router_revd_descol[0:2]))
router_revd_df.drop(router_revd_df.columns[cols_to_drop], axis=1, inplace=True)
router_revd_df.columns = df_cols
router_revd_df["Decision"] = router_revd_descol_value

print(router_revd_df.head())
print(router_revd_df.shape)

df_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size"]
cols_to_drop = [0,2,4,6]
router_sent_df = pd.read_csv(path+"TxSent_lrouter_0.csv")
router_sent_descol = router_sent_df.iloc[1][0].split(" ")
router_sent_descol_value = (" ".join(router_sent_descol[0:2]))
router_sent_df.drop(router_sent_df.columns[cols_to_drop], axis=1, inplace=True)
router_sent_df.columns = df_cols
router_sent_df["Decision"] = router_sent_descol_value
print(router_sent_df.head())
print(router_sent_df.shape)

router_drop_df.drop(columns=["Queue Size", "Packets dropped"], inplace=True)
router_revd_df.drop(columns=["Queue Size"], inplace=True)

test = pd.concat([router_drop_df, router_revd_df, router_sent_df], ignore_index=True)
test = test.sort_values(by=['Timestamp'], ascending=True)
test = test.reset_index(drop=True)
print(test.head())
test.to_csv(path+"combined_lrouter_0.csv")

numsenders = 1
currentsender = 0

while (currentsender < numsenders):
    sender_id = str(currentsender)
    sender_sent_df = pd.read_csv(path+"TxSent_sender_" + sender_id + ".csv")
    df_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size"]
    cols_to_drop = [0,2,4,6]
    sender_sent_descol = sender_sent_df.iloc[1][0].split(" ")
    sender_sent_descol_value = (" ".join(sender_sent_descol[0:2]))
    sender_sent_df.drop(sender_sent_df.columns[cols_to_drop], axis=1, inplace=True)
    sender_sent_df.columns = df_cols
    sender_sent_df["Decision"] = sender_sent_descol_value
    print(sender_sent_df.head())
    sender_sent_df.to_csv(path+"combined_sender" + sender_id + ".csv")
    currentsender+=1

print("preliminary test done...")