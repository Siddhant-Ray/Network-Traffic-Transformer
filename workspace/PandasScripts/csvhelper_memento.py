import os
import pandas as pd
import numpy as np
import argparse

print("Current directory is:", os.getcwd())
print("Generate combined csv for TCP congestion data")

def extract_TTL(text):
    list_of_features = text.split()
    idx_of_ttl = list_of_features.index("ttl")
    ttl = list_of_features[idx_of_ttl+1]
    return ttl

def extract_protocol(text):
    list_of_features = text.split()
    idx_of_protocol = list_of_features.index("protocol")
    protocol = list_of_features[idx_of_protocol+1]
    return protocol

def rename_flowid(input_text):
    return input_text

def generate_senders_csv(path, n_senders):

    path = path
    num_senders = n_senders
    sender_num = 0

    df_sent_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size",
                    "IP ID", "DSCP", "ECN", "TTL", "Payload Size", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port", "TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Delay", "Application ID", "Workload ID"]

    df_sent_cols_to_drop = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]

    temp_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size",
                    "IP ID", "DSCP", "ECN", "TTL", "Payload Size", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port", "TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Delay", "Application ID", "Workload ID"]
    
    temp = pd.DataFrame(columns = temp_cols)
    print(temp.head())

    files = ["memento_test10.csv", "memento_test20.csv", "memento_test25.csv"]

    for file in files:
        
        sender_tx_df = pd.read_csv(path+file)
        sender_tx_df = pd.DataFrame(np.vstack([sender_tx_df.columns,  sender_tx_df]))
        sender_tx_df.drop(sender_tx_df.columns[df_sent_cols_to_drop],axis=1,inplace=True)
        
        sender_tx_df.columns = df_sent_cols
        sender_tx_df["Packet ID"].iloc[0] = 0
        sender_tx_df["Flow ID"].iloc[0] = sender_tx_df["Flow ID"].iloc[1]
        sender_tx_df["IP ID"].iloc[0] = 0
        sender_tx_df["DSCP"].iloc[0] = 0
        sender_tx_df["ECN"].iloc[0] = 0
        sender_tx_df["TCP Sequence Number"].iloc[0] = 0
        # sender_tx_df["TTL"] = sender_tx_df.apply(lambda row: extract_TTL(row['Extra']), axis = 1)
        # sender_tx_df["Proto"] = sender_tx_df.apply(lambda row: extract_protocol(row['Extra']), axis = 1)
        sender_tx_df["Flow ID"] = [sender_num for i in range(sender_tx_df.shape[0])]

        df_sent_cols_new = ["Timestamp", "Flow ID", "Packet ID", "Packet Size",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Delay", "Application ID", "Workload ID"]
        sender_tx_df = sender_tx_df[df_sent_cols_new]
        
        # sender_tx_df.drop(['Extra'],axis = 1, inplace=True)
        temp = pd.concat([temp, sender_tx_df], ignore_index=True, copy = False)
        # sender_tx_df.drop(['Extra'],axis = 1, inplace=True)
        save_name = file.split(".")[0] + "_final.csv"
        sender_tx_df.to_csv(path+save_name, index=False)
    
    # temp.drop(['Extra'],axis = 1, inplace=True)
    print(temp.head())
    print(temp.columns)
    print(temp.shape)  

    return temp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod", "--model",
                        help = "choose CC model for creating congestion",
                        required = False)
    parser.add_argument("-nsend", "--numsenders",
                        help = "choose path for different topologies",
                        required = False)
    args = parser.parse_args()
    print(args)

    if args.model == "memento":
        path  = "/local/home/sidray/packet_transformer/outputs/results/"
    
    else:
        pass

    n_senders = 1
    sender_csv = generate_senders_csv(path, n_senders)
   
if __name__== '__main__':
    main()

