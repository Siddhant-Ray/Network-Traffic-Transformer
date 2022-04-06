import os
import pandas as pd
import numpy as np
import argparse

print("Current directory is:", os.getcwd())
print("Generate combined csv for TCP congestion data")

def generate_combined_csv(path, n_senders):

    path = path

    # Two senders create two interfaces on the router (only bottleneck router considered)
    num_router_intfs = 0 
    num_senders = n_senders

    df_drop_cols = ["Timestamp", "Queue Size", "Packets dropped", "Flow ID", "Packet ID", "Packet Size", "Interface ID"]
    df_drop_cols_to_drop = [0,2,4,6,8,10,12]

    df_revd_cols = ["Timestamp", "Queue Size", "Flow ID", "Packet ID", "Packet Size", "Interface ID"]
    df_revd_cols_to_drop = [0,2,4,6,8,10]

    df_sent_cols = ["Timestamp", "Flow ID", "Packet ID", "Packet Size", "Interface ID"]
    df_sent_cols_to_drop = [0,2,4,6,8]

    temp = pd.DataFrame(columns = df_sent_cols)

    while num_router_intfs < num_senders:

        router_drop_df = pd.read_csv(path+"RxDrops_lrouter_" + str(num_router_intfs) + ".csv")
        router_drop_df = pd.DataFrame(np.vstack([router_drop_df.columns, router_drop_df]))
        # print(router_drop_df.head())
        router_drop_descol = router_drop_df.iloc[1][0].split(" ")
        router_drop_descol_value = ( " ".join(router_drop_descol[0:2]))
        router_drop_df.drop(router_drop_df.columns[df_drop_cols_to_drop],axis=1,inplace=True)
        router_drop_df.columns = df_drop_cols
        router_drop_df["Decision"] = router_drop_descol_value
        # print(router_drop_df.head())
        # print(router_drop_df.shape)

        router_drop_df.drop(columns=["Queue Size", "Packets dropped"], inplace=True)

        temp = pd.concat([temp, router_drop_df], ignore_index=True, copy = False)
        
        router_revd_df = pd.read_csv(path+"RxRevd_lrouter_" + str(num_router_intfs) + ".csv")
        router_revd_df = pd.DataFrame(np.vstack([router_revd_df.columns, router_revd_df]))
        # print(router_revd_df.head())
        router_revd_descol = router_revd_df.iloc[1][0].split(" ")
        router_revd_descol_value = ( " ".join(router_revd_descol[0:2]))
        router_revd_df.drop(router_revd_df.columns[df_revd_cols_to_drop], axis=1, inplace=True)
        router_revd_df.columns = df_revd_cols
        router_revd_df["Decision"] = router_revd_descol_value

        # print(router_revd_df.head())
        # print(router_revd_df.shape)
        router_revd_df.drop(columns=["Queue Size"], inplace=True)

        temp = pd.concat([temp, router_revd_df], ignore_index=True)

        num_router_intfs += 1

    # print(temp.head())
    # print("Temp shape is ")
    # print(temp.shape)

    num_bottle_neckrouters = 1
    num_router_intfs = 0
    while num_router_intfs < num_bottle_neckrouters:

        router_sent_df = pd.read_csv(path+"TxSent_router_" + str(num_router_intfs) + ".csv")
        router_sent_df = pd.DataFrame(np.vstack([router_sent_df.columns, router_sent_df]))

        router_sent_descol = router_sent_df.iloc[1][0].split(" ")
        router_sent_descol_value = (" ".join(router_sent_descol[0:2]))
        router_sent_df.drop(router_sent_df.columns[df_sent_cols_to_drop], axis=1, inplace=True)
        router_sent_df.columns = df_sent_cols
        router_sent_df["Decision"] = router_sent_descol_value
        # print(router_sent_df.head())
        # print(router_sent_df.shape)

        temp = pd.concat([temp, router_sent_df], ignore_index=True)

        num_router_intfs += 1

    print(temp.head())
    print("Temp shape is ")
    print(temp.shape)

    temp['Timestamp'] = temp['Timestamp'].astype(float)

    final_router_info = temp.sort_values(by=['Timestamp'], ascending=True)
    final_router_info = final_router_info.reset_index(drop=True)
    print("Final router is: ")
    print(final_router_info.head())
    print(final_router_info.shape)

    return final_router_info

def generate_packet_delays_from_csv(input_dataframe):
    
    input_dataframe = input_dataframe[input_dataframe["Packet Size"].astype(float) >= 100]
    pack_ids = sorted(input_dataframe['Packet ID'].astype(float).unique())
 
    list_of_delay_ts = []
    for value in pack_ids:
        df_temp = input_dataframe.loc[input_dataframe['Packet ID'].astype(float) == float(value)]
        print(df_temp.tail())
        print(df_temp.shape)
        delay = df_temp
        break


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
    
    csv_to_save = generate_combined_csv(path, n_senders)
    csv_to_save.to_csv(path+"combined_router_0.csv")

    generate_packet_delays_from_csv(csv_to_save)

if __name__== '__main__':
    main()

