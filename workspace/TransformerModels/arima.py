from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from math import sqrt
import time as t
from datetime import datetime
import warnings
import pandas as pd, numpy as np
import argparse

from generate_sequences import generate_ARIMA_delay_data

NUMBOTTLECKS = 1

def run_arima():
    delay_data = generate_ARIMA_delay_data(NUMBOTTLECKS)
    targets, predictions = [], []
    warnings.simplefilter('ignore', ConvergenceWarning)

    # count = 0 
    # We want minimum 1023 for the first ARIMA prediction (size of the window)
    for value in range(1023, int(delay_data.shape[0]/116)+9990):

        # We want to predict the next value
        # Fit the model
        model = ARIMA(delay_data[:value], order=(1,1,2))
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
        df = pd.DataFrame({"Targets": targets, "Predictions": predictions.values})
        df.to_csv("/local/home/sidray/packet_transformer/evaluations/memento_data/ARIMA.csv", index=False)
    
    else:
        print("ARIMA load results from file")
        df = pd.read_csv("/local/home/sidray/packet_transformer/evaluations/memento_data/ARIMA.csv")

        targets = df["Targets"]
        predictions = df["Predictions"].str.split(" ").str[4].str.split("\n").str[0].astype(float)

        squared_error, mse = evaluate_arima(targets, predictions)
        
        df = pd.DataFrame({"Squared Error": squared_error, "targets": targets, "predictions": predictions})
        df.to_csv("/local/home/sidray/packet_transformer/evaluations/memento_data/ARIMA_evaluation.csv", index=False)

        print(df.head())

        print(squared_error.values)
        
        ## Stats on the squared error
        # Mean squared error
        print(np.mean(squared_error.values), " Mean squared error", )
        # Median squared error
        print(np.median(squared_error.values), " Median squared error")
        # 90th percentile squared error
        print(np.quantile(squared_error.values, 0.90, method = "closest_observation"), " 90th percentile squared error")
        # 99th percentile squared error
        print(np.quantile(squared_error.values, 0.99, method = "closest_observation"), " 99th percentile squared error")
        # 99.9th percentile squared error
        print(np.quantile(squared_error.values, 0.999, method = "closest_observation"), " 99.9th percentile squared error")
        # Standard deviation squared error
        print(np.std(squared_error.values), " Standard deviation squared error")
        
        ## Df row where the squared error is the a certain value
        print(df[df["Squared Error"] == np.quantile(squared_error.values, 0.5, method = "closest_observation")])
        print(df[df["Squared Error"] == np.quantile(squared_error.values, 0.90, method = "closest_observation")])
        print(df[df["Squared Error"] == np.quantile(squared_error.values, 0.99, method = "closest_observation")])
        print(df[df["Squared Error"] == np.quantile(squared_error.values, 0.999, method = "closest_observation")])
        print(df[df["Squared Error"] == np.quantile(squared_error.values, 0.9999, method = "closest_observation")]) 


