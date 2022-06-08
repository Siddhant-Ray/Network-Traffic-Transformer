import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot predictions')
    parser.add_argument('--path', type=str, help='Path to predictions files')
    parser.add_argument('--num_features', type=str, help='Number of input features')
    parser.add_argument('--window_size', type=str, help='Window size')
    
    args = parser.parse_args()

    if args.num_features == "16features" or args.num_features == "3features":
       path = args.path + "/" + args.num_features + "/"
    else:
        print("Choose valid number of features and path")
        exit()

    window_size = int(args.window_size)
    # Load data
    actual_values = np.load(path+"actual_last_delay_window_size_{}.npy".format(window_size))
    transformer_predictions = np.load(path+"transformer_last_delay_window_size_{}.npy".format(window_size))
    arma_predictions = np.load(path+"arma_last_delay_window_size_{}.npy".format(window_size))

    assert (actual_values.shape == transformer_predictions.shape == arma_predictions.shape)

    transformer_squared_losses = np.square(np.subtract(transformer_predictions, actual_values))
    arma_squared_losses = np.square(np.subtract(arma_predictions, actual_values))

    assert(transformer_squared_losses.shape == arma_squared_losses.shape)

    print("Mean loss on last delay from transformer is : ", np.mean(np.array(transformer_squared_losses)))
    print("Mean loss on last delay from arma is : ", np.mean(np.array(arma_squared_losses)))

    print("99%%ile loss on last delay from transformer is : ", np.percentile(np.array(transformer_squared_losses), 99))
    print("99%%ile loss on last delay from arma is : ", np.percentile(np.array(arma_squared_losses), 99))

    # Plot the actual delay 
    plt.figure()
    sns.histplot(actual_values)
    plt.legend([],[], frameon=False)
    plt.title("Actual delay distribution")
    plt.savefig("plots/actual_last_delay.png")

    # Plot the transformer predictions
    plt.figure()
    sns.histplot(transformer_predictions)
    plt.legend([],[], frameon=False)
    plt.title("Transformer predicted delay distribution")

    plt.savefig("plots/transformer_predictions.png")

    # Plot the arma predictions
    plt.figure()
    sns.histplot(arma_predictions)
    plt.legend([],[], frameon=False)
    plt.title("ARMA predicted delay distribution")
    plt.savefig("plots/arma_predictions.png")

    # Plot histograms of transformer and arma squared losses
    plt.figure()
    ax = sns.histplot(transformer_squared_losses, color="blue")
    ax.set(xlabel='Transformer Losses', ylabel='Frequency')
    plt.legend([],[], frameon=False)
    plt.title("Squared loss from transformer vs actual distribution")
    plt.savefig("plots/histogram_squared_losses_transformer.png")
    plt.figure()
    ax=sns.histplot(arma_squared_losses, color="red")
    ax.set(xlabel='ARMA Losses', ylabel='Frequency')
    plt.legend([],[], frameon=False)
    plt.title("Squared loss from ARMA vs actual distribution")
    plt.savefig("plots/histogram_squared_losses_arma.png")
    
if __name__=="__main__":
    main()
