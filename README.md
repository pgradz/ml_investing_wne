# ml_investing_wne

This repository was built with the aim of researching novel AI models in the context of investing.

## Setup
```
First, install dependencies. If training is done on macos, uncomment  tensorflow-macos and tensorflow-metal
    pip install -r requirements.txt

then install the package itself
    pip install .
```
## USAGE

The main entry point to the library is the **main_loop_ensemble.py** file. It trains deep learning models
in a loop over defined periods and averages results from different runs (seeds). It requires first to define config
parameters in config.py. Many of those parameters can be overwritten with command line arguments. Example of scripted
runs can be found in run_sensitivity.sh and run_rangebars.sh.

main_loop_ensemble_boosting.py is a modification of the main script to facilitate run for a classical machine learning
algorithm (XGBoost).

In order to access mlflow UI run:
```
mlflow.sh --folder-to-mlruns
```

All notebooks are non essential to the function of the library.

## DATA
Forex data can be found here https://www.histdata.com/
Crypto data can be found here https://www.binance.com/en/landing/data

This repository comes with processed data for ETHUSDT. Other cryptos can be reproduced
with code available in this repo. 
binance_data_processor.py was used to processed raw data from Binance
cumsum_filter.py was used to produced CUSUM and Range bars based on 1 min data


