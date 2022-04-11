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

    df_sent_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]

    df_sent_cols_to_drop = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]

    temp_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]
    
    temp = pd.DataFrame(columns = temp_cols)
    print(temp.head())

    while sender_num < num_senders:

        sender_tx_df = pd.read_csv(path+"TxSent_sender_" + str(sender_num) + ".csv")
        sender_tx_df = pd.DataFrame(np.vstack([sender_tx_df.columns,  sender_tx_df]))
        
        sender_tx_df.drop(sender_tx_df.columns[df_sent_cols_to_drop],axis=1,inplace=True)
        sender_tx_df.columns = df_sent_cols
        sender_tx_df["TTL"] = sender_tx_df.apply(lambda row: extract_TTL(row['Extra']), axis = 1)
        sender_tx_df["Proto"] = sender_tx_df.apply(lambda row: extract_protocol(row['Extra']), axis = 1)
        sender_tx_df["Flow ID"] = [sender_num for i in range(sender_tx_df.shape[0])]

        df_sent_cols_new = ["Timestamp", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]
        sender_tx_df = sender_tx_df[df_sent_cols_new]
        # sender_tx_df.drop(['Extra'],axis = 1, inplace=True)
        temp = pd.concat([temp, sender_tx_df], ignore_index=True, copy = False)

        sender_num += 1
   
    temp.drop(['Extra'],axis = 1, inplace=True)
    print(temp.head())
    print(temp.columns)
    print(temp.shape)  

    return temp

def generate_receivers_csv(path, n_receivers):

    path = path
    num_receivers = n_receivers
    receiver_num = 0

    df_revd_cols = ["Timestamp", "Queue Size", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]

    df_revd_cols_to_drop = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

    rtemp_cols = ["Timestamp", "Queue Size", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]
    
    rtemp = pd.DataFrame(columns = rtemp_cols)
    print(rtemp.head())

    while receiver_num < num_receivers:

        receiver_rx_df = pd.read_csv(path+"RxRevd_receiver_" + str(receiver_num) + ".csv")
        receiver_rx_df = pd.DataFrame(np.vstack([receiver_rx_df.columns,  receiver_rx_df]))
        receiver_rx_df.drop(receiver_rx_df.columns[df_revd_cols_to_drop],axis=1,inplace=True)
        receiver_rx_df.columns = df_revd_cols
        receiver_rx_df["TTL"] = receiver_rx_df.apply(lambda row: extract_TTL(row['Extra']), axis = 1)
        receiver_rx_df["Proto"] = receiver_rx_df.apply(lambda row: extract_protocol(row['Extra']), axis = 1)
        receiver_rx_df["Flow ID"] = [receiver_num for i in range(receiver_rx_df.shape[0])]

        df_revd_cols_new = ["Timestamp", "Queue Size", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Extra"]
        receiver_rx_df = receiver_rx_df[df_revd_cols_new]

        rtemp = pd.concat([rtemp, receiver_rx_df], ignore_index=True, copy = False)
        
        receiver_num += 1

    rtemp.drop(['Queue Size', 'Extra'],axis = 1, inplace=True)

    print(rtemp.head())
    print(rtemp.columns)
    print(rtemp.shape)  
   
    return rtemp


def generate_packet_delays_from_csv(input_dataframe1, input_dataframe2):
    
    input_dataframe1 = input_dataframe1[input_dataframe1["Packet Size"].astype(float) >= 100]
    input_dataframe2 = input_dataframe2[input_dataframe2["Packet Size"].astype(float) >= 100]
    
    pack_ids1 = sorted(input_dataframe1['Packet ID'].astype(float).unique())
    pack_ids2 = sorted(input_dataframe2['Packet ID'].astype(float).unique())
    print(len(pack_ids1))
    print(len(pack_ids2))
    pack_ids1.pop()
    assert(pack_ids1 == pack_ids2)
 
    list_of_delay_ts = []
    for value in pack_ids1:
        df_temp_s = input_dataframe1.loc[input_dataframe1['Packet ID'].astype(float) == float(value)]
        first_ts = df_temp_s.head(1)['Timestamp'].reset_index(drop=True)
    
        df_temp_r = input_dataframe2.loc[input_dataframe2['Packet ID'].astype(float) == float(value)]
        last_ts = df_temp_r.tail(1)['Timestamp'].reset_index(drop=True)

        # print(last_ts.iloc[0])
        # print(first_ts.iloc[0])

        delay_ts = last_ts.iloc[0] - first_ts.iloc[0]
        list_of_delay_ts.append(delay_ts)


    print(list_of_delay_ts)   
    print(len(list_of_delay_ts))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod", "--model",
                        help = "choose CC model for creating congestion",
                        required = True)
    parser.add_argument("-nsend", "--numsenders",
                        help = "choose path for different topologies",
                        required = True)
    args = parser.parse_args()
    print(args)

    if args.model == "tcponly":
        path  = "/local/home/sidray/packet_transformer/outputs/congestion_1/"
    elif args.model == "tcpandudp":
        path = "/local/home/sidray/packet_transformer/outputs/congestion_2/"
    else:
        print("ERROR: CONGESTION MODEL NOT CORRECT....")
        exit()

    n_senders = int(args.numsenders)
    n_receivers = n_senders

    sender_csv = generate_senders_csv(path, n_senders)
    receiver_csv = generate_receivers_csv(path, n_receivers)

    sender_csv.to_csv(path+"combined_sender.csv")
    
    generate_packet_delays_from_csv(sender_csv, receiver_csv)

if __name__== '__main__':
    main()

