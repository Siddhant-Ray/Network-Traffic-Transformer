from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from math import sqrt
import time as t
from datetime import datetime
import warnings
import pandas as pd

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

if __name__ == "__main__":
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
    df.to_csv("/local/home/sidray/packet_transformer/evaluations/memento_data/ARIMA.csv")