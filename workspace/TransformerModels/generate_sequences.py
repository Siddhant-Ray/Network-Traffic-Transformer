# Orignal author: Siddhant Ray

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import (
    convert_to_relative_timestamp, create_features_for_MCT, get_data_from_csv,
    ipaddress_to_number, make_windows_delay, make_windows_features,
    sliding_window_delay, sliding_window_features,
    vectorize_features_for_ARIMA, vectorize_features_to_numpy,
    vectorize_features_to_numpy_memento,
    vectorize_features_to_numpy_memento_with_receiver_IP_identifier)

# Params for the sliding window on the packet data
SLIDING_WINDOW_START = 0
SLIDING_WINDOW_STEP = 1
SLIDING_WINDOW_SIZE = 1024
WINDOW_BATCH_SIZE = 5000


def generate_sliding_windows(
    SLIDING_WINDOW_SIZE, WINDOW_BATCH_SIZE, num_features, TEST_ONLY_NEW, NUM_BOTTLENECKS
):

    sl_win_start = SLIDING_WINDOW_START
    sl_win_size = SLIDING_WINDOW_SIZE
    sl_win_shift = SLIDING_WINDOW_STEP

    num_features = num_features
    window_size = SLIDING_WINDOW_SIZE
    window_batch_size = WINDOW_BATCH_SIZE

    full_feature_arr = []
    full_target_arr = []
    test_loaders = []

    # Choose fine-tuning dataset (always True for this project)
    MEMENTO = True

    if MEMENTO:
        path = "memento_data/"

        if not TEST_ONLY_NEW:

            if NUM_BOTTLENECKS == 1:
                # These are the pre-training data files
                # These files were generated before the message ID feature was added, as that was not needed for pre-training.
                # The new files generated have the message ID added, just that the particular feature is not used for pre-training.
                # This is done to have simpler data generation, with all the features always for better consistency.
                # Thus, if data is newly generated for pre-training, we must replace
                # small_test_no_disturbance*_final.csv with small_test_no_disturbance_with_message_ids*_final.csv
                # where * is the seed number used for generating the data.
                files = [
                    "small_test_no_disturbance1_final.csv",
                    "small_test_no_disturbance2_final.csv",
                    "small_test_no_disturbance3_final.csv",
                    "small_test_no_disturbance4_final.csv",
                    "small_test_no_disturbance5_final.csv",
                    "small_test_no_disturbance6_final.csv",
                    "small_test_no_disturbance7_final.csv",
                    "small_test_no_disturbance8_final.csv",
                    "small_test_no_disturbance9_final.csv",
                    "small_test_no_disturbance10_final.csv",
                ]
            elif NUM_BOTTLENECKS == 2:
                files = [
                    "small_test_one_disturbance_with_message_ids1_final.csv",
                    "small_test_one_disturbance_with_message_ids2_final.csv",
                    "small_test_one_disturbance_with_message_ids3_final.csv",
                    "small_test_one_disturbance_with_message_ids4_final.csv",
                    "small_test_one_disturbance_with_message_ids5_final.csv",
                    "small_test_one_disturbance_with_message_ids6_final.csv",
                    "small_test_one_disturbance_with_message_ids7_final.csv",
                    "small_test_one_disturbance_with_message_ids8_final.csv",
                    "small_test_one_disturbance_with_message_ids9_final.csv",
                    "small_test_one_disturbance_with_message_ids10_final.csv",
                ]
            elif NUM_BOTTLENECKS == 4:  # Big topology
                files = [
                    "large_test_disturbance_with_message_ids1_final.csv",
                    "large_test_disturbance_with_message_ids2_final.csv",
                    "large_test_disturbance_with_message_ids3_final.csv",
                    "large_test_disturbance_with_message_ids4_final.csv",
                    "large_test_disturbance_with_message_ids5_final.csv",
                    "large_test_disturbance_with_message_ids6_final.csv",
                    "large_test_disturbance_with_message_ids7_final.csv",
                    "large_test_disturbance_with_message_ids8_final.csv",
                    "large_test_disturbance_with_message_ids9_final.csv",
                    "large_test_disturbance_with_message_ids10_final.csv",
                ]
            else:
                print("Invalid number of bottlenecks")
                exit()
        # Stale branch, never used
        else:
            files = ["small_test_one_disturbance2_final.csv"]
    # Stale branch, never used
    else:
        path = "congestion_1/"
        files = ["endtoenddelay_test.csv"]

    ## To calculate the global mean and std of the dataset
    global_df = pd.DataFrame(["Packet Size", "Delay"])
    for file in files:

        file_df = pd.read_csv(path + file)
        file_df = file_df[["Packet Size", "Delay"]]
        global_df = pd.concat([global_df, file_df], ignore_index=True)

    print(global_df.shape)
    mean_delay = global_df["Delay"].mean()
    std_delay = global_df["Delay"].std()
    mean_size = global_df["Packet Size"].mean()
    std_size = global_df["Packet Size"].std()

    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path + file)
        df = convert_to_relative_timestamp(df)
        df = ipaddress_to_number(df)
        df["Normalised Delay"] = df["Delay"].apply(
            lambda x: (x - mean_delay) / std_delay
        )
        df["Normalised Packet Size"] = df["Packet Size"].apply(
            lambda x: (x - mean_size) / std_size
        )

        if MEMENTO:
            if NUM_BOTTLENECKS == 1 or NUM_BOTTLENECKS == 2:
                feature_df, label_df = vectorize_features_to_numpy_memento(
                    df, reduced=True, normalize=True
                )
            elif NUM_BOTTLENECKS == 4:
                (
                    feature_df,
                    label_df,
                ) = vectorize_features_to_numpy_memento_with_receiver_IP_identifier(
                    df, reduced=True, normalize=True
                )
        else:
            feature_df, label_df = vectorize_features_to_numpy(df)

        print(feature_df.head(), feature_df.shape)
        print(label_df.head())

        # Create sliding window features
        input_array = np.hstack(feature_df.Combined.values.flatten())
        target_array = label_df.values
        feature_arr = list(
            make_windows_features(
                input_array, window_size, num_features, window_batch_size
            )
        )
        target_arr = list(
            make_windows_delay(target_array, window_size, window_batch_size)
        )

        ### OLD sliding window code
        # feature_arr = sliding_window_features(feature_df.Combined, sl_win_start, sl_win_size, sl_win_shift)
        # target_arr = sliding_window_delay(label_df, sl_win_start, sl_win_size, sl_win_shift)
        # print(len(feature_arr), len(target_arr))
        full_feature_arr = full_feature_arr + feature_arr
        full_target_arr = full_target_arr + target_arr

    print(len(full_feature_arr), len(full_target_arr))

    return full_feature_arr, full_target_arr, mean_delay, std_delay


def generate_MTC_data():
    full_feature_arr = []
    full_target_arr = []

    path = "memento_data/"
    files = ["small_test_one_disturbance_with_message_ids1_final.csv"]

    global_df = pd.DataFrame(["Packet Size", "Delay"])
    for file in files:

        file_df = pd.read_csv(path + file)
        file_df = file_df[["Packet Size", "Delay"]]
        global_df = pd.concat([global_df, file_df], ignore_index=True)

    print(global_df.shape)
    mean_delay = global_df["Delay"].mean()
    std_delay = global_df["Delay"].std()
    mean_size = global_df["Packet Size"].mean()
    std_size = global_df["Packet Size"].std()

    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path + file)
        df = convert_to_relative_timestamp(df)
        df = ipaddress_to_number(df)
        df["Normalised Delay"] = df["Delay"].apply(
            lambda x: (x - mean_delay) / std_delay
        )
        df["Normalised Packet Size"] = df["Packet Size"].apply(
            lambda x: (x - mean_size) / std_size
        )

        mct_df, mean_mct, std_mct, mean_msize, std_msize = create_features_for_MCT(
            df, reduced=True, normalize=True
        )

    mct_df.reset_index(drop=True, inplace=True)

    ## Remove the shape which are of incorrect transformer input type
    mct_df["shapes"] = [x.shape for x in mct_df["Input"].values]
    mct_df = mct_df[mct_df["shapes"] == (3072,)].drop("shapes", axis=1)

    return mct_df, mean_delay, std_delay, mean_mct, std_mct


def generate_ARIMA_delay_data(NUM_BOTTLENECKS):

    MEMENTO = True

    if MEMENTO:
        path = "memento_data/"

        if NUM_BOTTLENECKS == 1:
            files = ["small_test_no_disturbance1_final.csv"]
        elif NUM_BOTTLENECKS == 2:
            files = ["small_test_one_disturbance_with_message_ids1_final.csv"]
        elif NUM_BOTTLENECKS == 4:
            files = ["large_test_disturbance_with_message_ids1_final.csv"]

        else:
            print("Invalid number of bottlenecks")
            exit()

    else:
        path = "congestion_1/"
        files = ["endtoenddelay_test.csv"]

    for file in files:
        print(os.getcwd())

        df = get_data_from_csv(path + file)
        df = convert_to_relative_timestamp(df)
        df = ipaddress_to_number(df)

        label_df = vectorize_features_for_ARIMA(df)
        target_array = label_df

    return target_array


if __name__ == "__main__":

    delays = generate_ARIMA_delay_data(NUM_BOTTLENECKS=2)

    # generate_sliding_windows()
    final_df, mean_delay, std_delay, mean_mct, std_mct = generate_MTC_data()

    print(final_df)
    final_df.to_csv("memento_data/MCT.csv")

    print("Mean log size: ", np.mean(final_df["Log Message Size"]))
    print("90%ile log size: ", np.quantile(final_df["Log Message Size"], 0.90))
    print("99%ile log size: ", np.quantile(final_df["Log Message Size"], 0.99))
    print("99.9%ile log size: ", np.quantile(final_df["Log Message Size"], 0.999))

    print("Mean log MCT: ", np.mean(final_df["Log Message Completion Time"]))
    print(
        "90%ile log MCT: ", np.quantile(final_df["Log Message Completion Time"], 0.90)
    )
    print(
        "99%ile log MCT: ", np.quantile(final_df["Log Message Completion Time"], 0.99)
    )
    print(
        "99.9%ile log MCT: ",
        np.quantile(final_df["Log Message Completion Time"], 0.999),
    )

    print("Mean packet count: ", np.mean(final_df["Packet Count"]))
    print("90%ile packet count: ", np.quantile(final_df["Packet Count"], 0.90))
    print("99%ile packet count: ", np.quantile(final_df["Packet Count"], 0.99))

    print("Mean MCT: ", np.mean(final_df["Message Completion Time"]))
    print("90%ile MCT: ", np.quantile(final_df["Message Completion Time"], 0.90))
    print("99%ile MCT: ", np.quantile(final_df["Message Completion Time"], 0.99))
    print("99.9%ile MCT: ", np.quantile(final_df["Message Completion Time"], 0.999))
    print("99.99%ile MCT: ", np.quantile(final_df["Message Completion Time"], 0.9999))

    plt.figure()

    sbs = sns.displot(data=final_df, kind="ecdf", x="Normalised Log Message Size")

    sbs.fig.suptitle("Log Normalised Message Size")
    plt.savefig("Norm_message_size" + ".png")

    plt.figure()

    sbs = sns.displot(data=final_df, kind="ecdf", x="Normalised Log MCT")

    sbs.fig.suptitle("Log Normalised MCT")
    plt.savefig("Norm_MCT" + ".png")

    plt.figure()
    sbs = sns.displot(data=final_df, kind="ecdf", x="Log Message Completion Time")

    sbs.fig.suptitle("Log Message Completion Time")
    plt.savefig("MCT" + ".png")

    plt.figure()

    sbs = sns.displot(data=final_df, kind="ecdf", x="Log Message Size")

    sbs.fig.suptitle("Log Message size")
    plt.savefig("Message_size" + ".png")
