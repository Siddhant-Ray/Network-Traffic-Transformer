# Orignal author: Siddhant Ray

import argparse,os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ewma(seq, alpha=1):
    w_new = alpha
    w_old = 1 - alpha

    output = [seq[0]]
    old_val = seq[0]
    for new_val in seq[1:]:
        old_val = w_new * new_val + w_old * old_val
        output.append(old_val)
    return np.array(output)


def mse(seq_a, seq_b):
    return np.mean((seq_a - seq_b) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Plot predictions")
    parser.add_argument("--path", type=str, help="Path to predictions files")
    parser.add_argument("--num_features", type=str, help="Number of input features", default="3features")
    parser.add_argument("--window_size", type=str, help="Window size")

    args = parser.parse_args()

    if args.num_features == "16features" or args.num_features == "3features":
        print("Plotting for {} features".format(args.num_features))
        path = args.path + "/"
    else:
        print("Choose valid number of features and path")
        exit()

    window_size = int(args.window_size)
    # Load data
    actual_values = np.load(
        path + "actual_last_delay_window_size_{}.npy".format(window_size)
    )
    transformer_predictions = np.load(
        path + "transformer_last_delay_window_size_{}.npy".format(window_size)
    )
    arma_predictions = np.load(
        path + "arma_last_delay_window_size_{}.npy".format(window_size)
    )
    penultimate_predictions = np.load(
        path + "penultimate_last_delay_window_size_{}.npy".format(window_size)
    )
    ewm_predictions = np.load(path + "ewm_delay_window_size_{}.npy".format(window_size))

    assert (
        actual_values.shape == transformer_predictions.shape == arma_predictions.shape
    )
    assert actual_values.shape == penultimate_predictions.shape == ewm_predictions.shape

    transformer_squared_losses = np.square(
        np.subtract(transformer_predictions, actual_values)
    )
    arma_squared_losses = np.square(np.subtract(arma_predictions, actual_values))
    penultimate_squared_losses = np.square(
        np.subtract(penultimate_predictions, actual_values)
    )
    ewm_squared_losses = np.square(np.subtract(ewm_predictions, actual_values))

    assert transformer_squared_losses.shape == arma_squared_losses.shape

    print(
        "Mean loss on last delay from transformer is : ",
        np.mean(np.array(transformer_squared_losses)),
    )
    print(
        "99%%ile loss on last delay from transformer is : ",
        np.percentile(np.array(transformer_squared_losses), 99),
    )

    print(
        "Mean loss on last delay from arma is : ",
        np.mean(np.array(arma_squared_losses)),
    )
    print(
        "99%%ile loss on last delay from arma is : ",
        np.percentile(np.array(arma_squared_losses), 99),
    )

    print(
        "Mean loss on last delay from ewma is : ", np.mean(np.array(ewm_squared_losses))
    )
    print(
        "99%%ile loss on last delay from ewma is : ",
        np.percentile(np.array(ewm_squared_losses), 99),
    )

    if args.path == "plot_values":
        median_predictions = np.load(
            path + "median_last_delay_window_size_{}.npy".format(window_size)
        )
        assert actual_values.shape == median_predictions.shape

        median_squared_losses = np.square(
            np.subtract(median_predictions, actual_values)
        )

        print(
            "Mean loss on last delay from median is : ",
            np.mean(np.array(median_squared_losses)),
        )
        print(
            "99%%ile loss on last delay from median is : ",
            np.percentile(np.array(median_squared_losses), 99),
        )

    # ## Print 20 values of each with high precision
    np.set_printoptions(precision=32)
    print("Actual values are : ", actual_values[:20])
    print("Transformer predictions are : ", transformer_predictions[:20])
    print("ARMA predictions are : ", arma_predictions[:20])
    print("EWM predictions are : ", ewm_predictions[:20])

    ## Save path
    if not os.path.exists(path + "plots/"):
        os.makedirs(path + "plots/")
    save_path = path + "plots/"
    
    # Reshape into 1D arrays for seaborn plots
    actual_values = actual_values.reshape(-1)
    transformer_predictions = transformer_predictions.reshape(-1)
    arma_predictions = arma_predictions.reshape(-1)
    penultimate_predictions = penultimate_predictions.reshape(-1)
    ewm_predictions = ewm_predictions.reshape(-1)

    # Print 100 values of each with high precision
    np.set_printoptions(precision=32)
    # print("Actual values are : ", actual_values[:100])
    # print("Transformer predictions are : ", transformer_predictions[:100])
    # print("ARMA predictions are : ", arma_predictions[:100])
    # print("EWM predictions are : ", ewm_predictions[:100])

    # Plot actual IAT vs transformer predicted delay
    # # Plot first 1000 values
    # plt.figure()
    # sns.lineplot(
    #     x=np.arange(1000),
    #     y=actual_values[:1000],
    #     label="Actual delay",
    #     color="blue",
    # )
    # sns.lineplot(
    #     x=np.arange(1000),
    #     y=transformer_predictions[:1000],
    #     label="Transformer predictions",
    #     color="orange",
    # )
    # plt.legend()
    # #plt.ylim(2e-6, 4e-6)
    # plt.title("Actual delay vs transformer predicted delay")
    # plt.savefig(save_path + "transformer_vs_actual_predictions.pdf")

    # Make a subplot of delay vs transformer predictions 
    # with first 10, 50, 100, 200, 500, 1000 predictions
    # Don't show legend for each subplot

    # Calculate entropy of the actual values distribution
    # use scipy.stats.entropy
    # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

    # from scipy.stats import entropy
    # entropy = entropy(actual_values)
    # print("Entropy of actual values is : ", entropy)

    target = args.path.split("_")[-1]

    # Plot actual values vs transformer predictions
    plt.figure()
    plt.subplot(3, 3, 1)
    sns.lineplot(
        x=np.arange(10),
        y=actual_values[:10],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(len(transformer_predictions)),
        y=transformer_predictions,
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title("10 predictions")

    plt.subplot(3, 3, 2)
    sns.lineplot(
        x=np.arange(25),
        y=actual_values[:25],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(25),
        y=transformer_predictions[:25],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title(f"25 predictions")

    plt.subplot(3, 3, 3)
    sns.lineplot(
        x=np.arange(50),
        y=actual_values[:50],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(50),
        y=transformer_predictions[:50],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title(f"50 predictions")


    plt.subplot(3, 3, 4)
    sns.lineplot(
        x=np.arange(75),
        y=actual_values[:75],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(75),
        y=transformer_predictions[:75],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title(f"75 predictions")

    plt.subplot(3, 3, 5)
    sns.lineplot(
        x=np.arange(100),
        y=actual_values[:100],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(100),
        y=transformer_predictions[:100],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title(f"100 predictions")

    plt.subplot(3, 3, 6)
    sns.lineplot(
        x=np.arange(200),
        y=actual_values[:200],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(200),
        y=transformer_predictions[:200],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()

    plt.title(f"200 predictions")

    plt.subplot(3, 3, 7)
    sns.lineplot(
        x=np.arange(300),
        y=actual_values[:300],
        label=f"Actual {target}",
        color="blue",
    )

    sns.lineplot(
        x=np.arange(300),
        y=transformer_predictions[:300],
        label="Transformer predictions",
        color="orange",
    )

    plt.legend()
    plt.title(f"300 predictions")

    plt.subplot(3, 3, 8)
    sns.lineplot(
        x=np.arange(500),
        y=actual_values[:500],
        label=f"Actual {target}",
        color="blue",
    )

    sns.lineplot(
        x=np.arange(500),
        y=transformer_predictions[:500],
        label="Transformer predictions",
        color="orange",
    )

    plt.legend()

    plt.title(f"500 predictions")

    # Remove legend for each subplot
    for ax in plt.gcf().axes:
        try:
            ax.legend_.remove()
        except:
            pass

    plt.suptitle(f"Actual {target} vs transformer predicted {target}", fontsize=8)
    # Add only one legend for the entire figure
    # set line colors for legend as blue and orange
    # Match the legend labels to the line colors from the subplots

    plt.figlegend(
        labels=["Actual delay", "Transformer predictions"],
        loc="lower center",
        ncol=2,
        fontsize=8,
        labelcolor=["blue", "orange"],
    )
    
    plt.tight_layout()

    plt.savefig(save_path + f"transformer_vs_actual_predictions_{target}.pdf")
    exit()

    plt.figure()
    # Make figure share y axis
    # plt.subplots(sharey=True, figsize=(10, 10))
    plt.subplot(3, 3, 1)
    sns.lineplot(
        x=np.arange(10),
        y=actual_values[:10],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(10),
        y=transformer_predictions[:10],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 10)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("10 predictions", fontsize=6)

    plt.subplot(3, 3, 2)
    sns.lineplot(
        x=np.arange(25),
        y=actual_values[:25],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(25),
        y=transformer_predictions[:25],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 25)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("25 predictions", fontsize=6)

    plt.subplot(3, 3, 3)
    sns.lineplot(
        x=np.arange(50),
        y=actual_values[:50],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(50),
        y=transformer_predictions[:50],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 50)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("50 predictions", fontsize=6)

    plt.subplot(3, 3, 4)
    sns.lineplot(
        x=np.arange(75),
        y=actual_values[:75],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(75),
        y=transformer_predictions[:75],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 75)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("75 predictions", fontsize=6)

    plt.subplot(3, 3, 5)
    sns.lineplot(
        x=np.arange(100),
        y=actual_values[:100],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(100),
        y=transformer_predictions[:100],
        label="Transformer predictions",
        color="orange",
    )
    plt.legend(fontsize=6, loc="upper right")
    #plt.ylim(0, 0.4)
    plt.xlim(0, 100)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.title("100 predictions", fontsize=6)

    plt.subplot(3, 3, 6)
    sns.lineplot(
        x=np.arange(200),
        y=actual_values[:200],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(200),
        y=transformer_predictions[:200],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 200)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("200 predictions", fontsize=6)

    plt.subplot(3, 3, 7)
    sns.lineplot(
        x=np.arange(300),
        y=actual_values[:300],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(300),
        y=transformer_predictions[:300],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 300)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("300 predictions", fontsize=6)

    plt.subplot(3, 3, 8)
    sns.lineplot(
        x=np.arange(500),
        y=actual_values[:500],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(500),
        y=transformer_predictions[:500],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 500)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("500 predictions", fontsize=6)
    
    plt.subplot(3, 3, 9)
    sns.lineplot(
        x=np.arange(1000),
        y=actual_values[:1000],
        label=f"Actual {target}",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(1000),
        y=transformer_predictions[:1000],
        label="Transformer predictions",
        color="orange",
    )
    #plt.ylim(0, 0.4)
    plt.xlim(0, 1000)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.legend(fontsize=6, loc="upper right")
    plt.title("1000 predictions", fontsize=6)
    
    # Remove legend for each subplot
    for ax in plt.gcf().axes:
        try:
            ax.legend_.remove()
        except:
            pass

    plt.suptitle(f"Actual {target} vs transformer predicted {target}", fontsize=8)
    # Add only one legend for the entire figure
    # set line colors for legend as blue and orange
    # Match the legend labels to the line colors from the subplots

    plt.figlegend(
        labels=["Actual delay", "Transformer predictions"],
        loc="lower center",
        ncol=2,
        fontsize=8,
        labelcolor=["blue", "orange"],
    )
    
    plt.tight_layout()
    plt.savefig(save_path + f"transformer_vs_actual_predictions_subplots_{target}.pdf")
    
    # Plot actual IAT vs EWM predicted delay
    # Plot first 1000 values
    plt.figure()
    sns.lineplot(
        x=np.arange(1000),
        y=actual_values[:1000],
        label="Actual delay",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(1000),
        y=ewm_predictions[:1000],
        label="EWM predictions",
        color="orange",
    )
    plt.legend()
    plt.title("Actual delay vs EWM predicted delay")
    plt.savefig(save_path + "ewm_vs_actual_predictions.pdf")

    # Plot histogram of actual delay values vs transformer predicted 
    # delay values
    plt.figure()
    plt.hist(actual_values, bins=np.linspace(0, 0.5, 101),
                label = "Actual delays",  alpha = 0.4)
    plt.hist(transformer_predictions, bins=np.linspace(0, 0.5, 101),
                label = "Transformer predictions",  alpha = 0.6, 
                histtype='step', color = 'blue')

    plt.legend()
    plt.title("Actual delay vs transformer predicted delay distribution")
    plt.savefig(save_path + "histogram_transformer_vs_actual_predictions.pdf")

    # Plot histogram of actual delay values vs ewm predicted
    # delay values
    plt.figure()
    plt.hist(actual_values, bins=np.linspace(0, 0.5, 101),
                label = "Actual delays",  alpha = 0.4)
    plt.hist(ewm_predictions, bins=np.linspace(0, 0.5, 101),
                label = "EWM predictions",  alpha = 0.6, 
                histtype='step', color = 'blue')
    
    plt.legend()
    plt.title("Actual delay vs EWM predicted delay distribution")
    plt.savefig(save_path + "histogram_ewm_vs_actual_predictions.pdf")

    print()
    print()

    # print 99%ile values for each
    print("99%ile actual delay value is : ", np.percentile(actual_values, 99))
    print(
        "99%ile transformer predicted delay value is : ",
        np.percentile(transformer_predictions, 99),
    )
    print(
        "99%ile arma predicted delay value is : ",
        np.percentile(arma_predictions, 99),
    )
    print(
        "99%ile ewm predicted delay value is : ",
        np.percentile(ewm_predictions, 99),
    )

    # Get mean of all delay values
    mean_delay = np.mean(actual_values)
    median_delay = np.median(actual_values)

    print("Mean delay is : ", mean_delay)
    print("Median delay is : ", median_delay)

    # Print minimum and maximum values
    print("Minimum delay is : ", np.min(actual_values))
    print("Maximum delay is : ", np.max(actual_values))

    exit()

    plt.figure()
    sns.lineplot(
        x=np.arange(len(actual_values)),
        y=actual_values,
        label="Actual delay",
        color="blue",
    )
    sns.lineplot(
        x=np.arange(len(transformer_predictions)),
        y=transformer_predictions,
        label="Transformer predictions",
        color="orange",
    )
    plt.legend()
    plt.title("Actual IAT vs transformer predicted IAT")
    plt.savefig(save_path + "transformer_vs_actual_predictions.pdf")

    input_seq = actual_values
    target = input_seq[1:]
    predictions = input_seq[:-1]

    smoothed_001 = ewma(
        predictions, alpha=0.01
    )  # This should equal our current res. (updated!)
    # Some extras I'd like to try.
    smoothed_01 = ewma(predictions, alpha=0.1)
    smoothed_05 = ewma(predictions, alpha=0.5)
    smoothed_09 = ewma(predictions, alpha=0.9)

    # Normally compute MSE.
    mse_unsmoothed = mse(target, predictions)
    mse_001 = mse(target, smoothed_001)
    mse_01 = mse(target, smoothed_01)
    mse_05 = mse(target, smoothed_05)
    mse_09 = mse(target, smoothed_09)

    print("Unsmoothed", mse_unsmoothed)
    print("Alpha 0.01", mse_001)
    print("Alpha 0.1", mse_01)
    print("Alpha 0.5", mse_05)
    print("Alpha 0.9", mse_09)


if __name__ == "__main__":
    main()
