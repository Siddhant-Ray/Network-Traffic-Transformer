# Orignal author: Siddhant Ray

import argparse
import time as t
import warnings
from datetime import datetime
from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from generate_sequences import generate_ARIMA_delay_data
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

NUMBOTTLECKS = 1


def run_arima():
    delay_data = generate_ARIMA_delay_data(NUMBOTTLECKS)
    targets, predictions = [], []
    warnings.simplefilter("ignore", ConvergenceWarning)

    # count = 0
    # We want minimum 1023 for the first ARIMA prediction (size of the window)
    # Make this 29990 -> 9990 for the 10000 history window ARIMA
    for value in range(1023, int(delay_data.shape[0] / 116) + 29990):

        # We want to predict the next value
        # Fit the model
        model = ARIMA(delay_data[:value], order=(1, 1, 2))
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)
        targets.append(delay_data[value])
        predictions.append(yhat)
        # count+=1

    # print(count)
    return targets, predictions


def evaluate_arima(targets, predictions):
    mse = mean_squared_error(targets, predictions)
    squared_error = np.square(targets - predictions)
    return squared_error, mse


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--run", type=bool, default=False)
    args = args.parse_args()

    if args.run:

        print("Started ARIMA at:")
        time = datetime.now()
        print(time)

        targets, predictions = run_arima()

        ## MSE calculation
        mse = mean_squared_error(targets, predictions)
        print(mse)

        print("Finished ARIMA at:")
        time = datetime.now()
        print(time)

        # Save the results
        df = pd.DataFrame({"Targets": targets, "Predictions": predictions})
        df.to_csv("memento_data/ARIMA_30000.csv", index=False)

    else:
        print("ARIMA load results from file")
        df = pd.read_csv("memento_data/ARIMA_30000.csv")

        targets = df["Targets"]
        predictions = (
            df["Predictions"].str.split(" ").str[4].str.split("\n").str[0].astype(float)
        )

        squared_error, mse = evaluate_arima(targets, predictions)

        df = pd.DataFrame(
            {
                "Squared Error": squared_error,
                "targets": targets,
                "predictions": predictions,
            }
        )
        df.to_csv("memento_data/ARIMA_evaluation_30000.csv", index=False)

        print(df.head())

        print(squared_error.values)

        ## Stats on the squared error
        # Mean squared error
        print(
            np.mean(squared_error.values),
            " Mean squared error",
        )
        # Median squared error
        print(np.median(squared_error.values), " Median squared error")
        # 90th percentile squared error
        print(
            np.quantile(squared_error.values, 0.90, method="closest_observation"),
            " 90th percentile squared error",
        )
        # 99th percentile squared error
        print(
            np.quantile(squared_error.values, 0.99, method="closest_observation"),
            " 99th percentile squared error",
        )
        # 99.9th percentile squared error
        print(
            np.quantile(squared_error.values, 0.999, method="closest_observation"),
            " 99.9th percentile squared error",
        )
        # Standard deviation squared error
        print(np.std(squared_error.values), " Standard deviation squared error")

        ## Df row where the squared error is the a certain value
        print(
            df[
                df["Squared Error"]
                == np.quantile(squared_error.values, 0.5, method="closest_observation")
            ],
            "Values at median SE",
        )
        print(
            df[
                df["Squared Error"]
                == np.quantile(squared_error.values, 0.90, method="closest_observation")
            ],
            "Values at 90th percentile SE",
        )
        print(
            df[
                df["Squared Error"]
                == np.quantile(squared_error.values, 0.99, method="closest_observation")
            ],
            "Values at 99th percentile SE",
        )
        print(
            df[
                df["Squared Error"]
                == np.quantile(
                    squared_error.values, 0.999, method="closest_observation"
                )
            ],
            "Values at 99.9th percentile SE",
        )
        print(
            df[
                df["Squared Error"]
                == np.quantile(
                    squared_error.values, 0.9999, method="closest_observation"
                )
            ],
            "Values at 99.99th percentile SE",
        )

        # Plot the index vs squared error
        # Set figure size

        sns.set_theme("paper", "whitegrid", font_scale=1.5)
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                "lines.linewidth": 2,
                "lines.markeredgewidth": 0,
                "scatter.marker": ".",
                "scatter.edgecolors": "none",
                # Set image quality and reduce whitespace around saved figure.
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.01,
            }
        )

        # Plot the index vs squared error
        plt.figure(figsize=(5, 5))
        sns.lineplot(x=df.index, y=df["Squared Error"], color="red")
        plt.xlabel("History Length")
        plt.ylabel("Squared Error")
        # plt.title("Squared Error on predictions vs History Length upto 10000")
        plt.xlim(1023, 10000)
        # place legend to the right, bbox is the box around the legend
        # plt.legend(loc='upper right', bbox_to_anchor=(0.67, 1.02), ncol=1)
        plt.savefig("SE_trend_arima_10000.pdf")

        ## Do the plots over a loop of xlims
        xlims = [0, 6000, 12000, 18000, 24000, 30000]
        for idx_xlim in range(len(xlims) - 1):
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                x=df.index, y=df["Squared Error"], color="red", label="Squared Error"
            )
            # label axes
            plt.xlabel("History Length")
            plt.ylabel("Squared Error")
            # set xlim
            plt.xlim(xlims[idx_xlim], xlims[idx_xlim + 1])
            plt.title(
                "Squared Error trend for xlims "
                + str(xlims[idx_xlim])
                + " to "
                + str(xlims[idx_xlim + 1])
            )
            plt.savefig(
                "SE_trend_arima_xlim_"
                + str(xlims[idx_xlim])
                + "_"
                + str(xlims[idx_xlim + 1])
                + ".pdf"
            )
