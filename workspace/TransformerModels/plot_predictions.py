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
    penultimate_predictions = np.load(path+"penultimate_last_delay_window_size_{}.npy".format(window_size))
    ewm_predictions = np.load(path+"ewm_delay_window_size_{}.npy".format(window_size))
    
    assert (actual_values.shape == transformer_predictions.shape == arma_predictions.shape)
    assert (actual_values.shape == penultimate_predictions.shape == ewm_predictions.shape)

    transformer_squared_losses = np.square(np.subtract(transformer_predictions, actual_values))
    arma_squared_losses = np.square(np.subtract(arma_predictions, actual_values))
    penultimate_squared_losses = np.square(np.subtract(penultimate_predictions, actual_values))
    ewm_squared_losses = np.square(np.subtract(ewm_predictions, actual_values))

    assert(transformer_squared_losses.shape == arma_squared_losses.shape)

    print("Mean loss on last delay from transformer is : ", np.mean(np.array(transformer_squared_losses)))
    print("99%%ile loss on last delay from transformer is : ", np.percentile(np.array(transformer_squared_losses), 99))

    print("Mean loss on last delay from arma is : ", np.mean(np.array(arma_squared_losses)))
    print("99%%ile loss on last delay from arma is : ", np.percentile(np.array(arma_squared_losses), 99))

    print("Mean loss on last delay from penultimate is : ", np.mean(np.array(penultimate_squared_losses)))
    print("99%%ile loss on last delay from penultimate is : ", np.percentile(np.array(penultimate_squared_losses), 99))

    print("Mean loss on last delay from ewma is : ", np.mean(np.array(ewm_squared_losses)))
    print("99%%ile loss on last delay from ewma is : ", np.percentile(np.array(ewm_squared_losses), 99))

    if args.path == "plot_values":
        median_predictions = np.load(path+"median_last_delay_window_size_{}.npy".format(window_size))
        assert(actual_values.shape == median_predictions.shape)

        median_squared_losses = np.square(np.subtract(median_predictions, actual_values))

        print("Mean loss on last delay from median is : ", np.mean(np.array(median_squared_losses)))
        print("99%%ile loss on last delay from median is : ", np.percentile(np.array(median_squared_losses), 99))

    ## Save path 
    save_path = "plots/" + path 

    # Plot the actual delay vs transformer predictions 
    '''plt.figure()
    plt.hist(actual_values, bins=np.linspace(0, 0.5, 101), label = "Actual delays",  alpha = 0.4)
    plt.hist(transformer_predictions, bins=np.linspace(0, 0.5, 101), label = "Transformer predictions",  alpha = 0.6, histtype='step', color = 'blue')
    
    plt.legend()
    plt.title("Actual delay vs transformer predicted delay distribution")
    # plt.savefig(save_path + "transformer_vs_actual_predictions.png")

    # Plot the actual delay vs arma predictions
    plt.figure()
    plt.hist(actual_values, bins=np.linspace(0, 0.5, 101), label = "Actual delays", alpha = 0.4)
    plt.hist(arma_predictions, bins=np.linspace(0, 0.5, 101), label = "ARMA predictions", alpha = 0.6, histtype='step', color = 'blue')
    
    plt.legend()
    plt.title("Actual delay vs arma predicted delay distribution")
    # plt.savefig(save_path + "arma_vs_actual_predictions.png")'''

    plt.figure()
    # plt.hist(actual_values, bins=np.linspace(0, 0.5, 101), label = "Actual delays",  alpha = 0.8, histtype='step', color = 'blue', linewidth=2)
    plt.hist(transformer_predictions, bins=np.linspace(0, 0.5, 101), label = "Transformer predictions",  alpha = 0.6, color = 'orange')
    plt.hist(ewm_predictions, bins=np.linspace(0, 0.5, 101), label = "EWMA predictions",  alpha = 0.2, color = 'green')
    plt.legend()
    plt.title("Transformer vs baselines predicted delay distribution")
    plt.savefig(save_path + "transformer_vs_baseline_predictions.png")

    # Plot histograms of transformer and arma squared losses
    plt.figure()
    ax = sns.histplot(transformer_squared_losses, color="blue")
    ax.set(xlabel='Transformer Losses', ylabel='Frequency')
    plt.legend([],[], frameon=False)
    plt.title("Squared loss from transformer vs actual distribution")
    plt.xlim(0, 0.0005)
    plt.savefig(save_path + "histogram_squared_losses_transformer.png")

    plt.figure()
    ax=sns.histplot(arma_squared_losses, color="red")
    ax.set(xlabel='ARMA Losses', ylabel='Frequency')
    plt.legend([],[], frameon=False)
    plt.title("Squared loss from ARMA vs actual distribution")
    plt.xlim(0, 0.0005)
    plt.savefig(save_path + "histogram_squared_losses_arma.png")
    
if __name__=="__main__":
    main()
