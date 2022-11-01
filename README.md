# ml_investing_wne

This repository was built with the aim of researching novel AI models in the context of investing.

## Setup
First, install dependencies
```
    pip install -r requirements.txt
```
then install the package itself
```
    pip install ml_investing_wne -e .
```
## USAGE

The main entry point to the library is the **main.py** file. It requires first to define config
parameters in config.py. Upon execution, it will train and save selected deep learning model and store
performance metrics using mlflow.

In order to access mlflow UI run:
```
mlflow.sh --folder-to-mlruns
```

**Trading** functionalties are provided within xtb submodule named after platform selected 
for real life testing. In order to use it, it requires account creation.

## DATA
Most of the data used in the research thus far come from https://www.histdata.com/

