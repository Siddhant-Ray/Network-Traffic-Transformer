import os
import pandas as pd
import numpy as np
import argparse, collections

print("Current directory is:", os.getcwd())
print("Generate end to end packet delays")

## Some packets are sent at sender but not received at receiver
## they are dropped somewhere in the middle, so those are not counted
## for the delay calculation 
def gen_packet_delay(input_dataframe1, input_dataframe2, path):
    
    # print(input_dataframe1.head(), input_dataframe1.shape)
    # print(input_dataframe2.head(), input_dataframe2.shape)

    ## these packets are not received
    ids_not_received = list(set(input_dataframe1['IP ID'].to_list()) - 
                        set(input_dataframe2['IP ID'].to_list()))

    # print(ids_not_received)

    for value in ids_not_received:
        input_dataframe1 = input_dataframe1[input_dataframe1["IP ID"] != value]
    input_dataframe1 = input_dataframe1.reset_index(drop=True)

    # print(input_dataframe1.tail(), input_dataframe1.shape)
    # print(input_dataframe2.tail(), input_dataframe2.shape)

    input_dataframe1["Delay"] = input_dataframe2["Timestamp"] - input_dataframe1["Timestamp"]
    # print(input_dataframe1.tail(), input_dataframe1.shape)
    
    return  input_dataframe1

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

    num_senders = int(args.numsenders)
    num_receivers = num_senders
    sender = 0 
    receiver = 0 

    temp_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size", "Interface ID",
                    "IP ID", "DSCP","ECN", "Payload Size", "TTL", "Proto", "Source IP", "Destination IP",
                    "TCP Source Port","TCP Destination Port", "TCP Sequence Number", "TCP Window Size",
                    "Delay"]

    temp = pd.DataFrame(columns = temp_cols)

    while sender < num_senders and receiver < num_receivers:

        input_dataframe1 = pd.read_csv(path+"sender_{}_final.csv".format(sender))
        input_dataframe2 = pd.read_csv(path+"receiver_{}_final.csv".format(receiver))

        delay_df = gen_packet_delay(input_dataframe1,input_dataframe2, path)
        temp = pd.concat([temp, delay_df], ignore_index=True, copy = False)

        sender+=1
        receiver+=1
   
    temp = temp.sort_values(by=['Timestamp'], ascending=True) 
    print(temp.head())  
    print(temp.shape)
    temp.to_csv(path+"endtoenddelay.csv", index=False)

if __name__== '__main__':
    main()
