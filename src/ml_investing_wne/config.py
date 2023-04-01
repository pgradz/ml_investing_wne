import os
import datetime


RUN_TYPE = 'forex' # forex or crypto
RUN_SUBTYPE = 'time_aggregated' #'triple_barrier_time_aggregated','time_aggregated', 'volume_bars'
provider = 'hist_data' # hist_data, Bitstamp, Binance

currency = 'EURCHF'
# leave empty if training from scratch, for transfer learning specify currency to be used as a base
load_model = ''
freq = '720min'
input_dim = '1d'  # 2d or 1d
# has to be defined inside tf_models folder
model = 'transformer_learnable_encoding' # resnet_lstm_regularized, transformer_learnable_encoding, lstm
seed = 12345
# volume for volume bars
volume = 50000



# Tripple barrier method
t_final=24
fixed_barrier=0.001

# Cost is expressed in pips
COST_FOREX = {
    'EURCHF': 2,
    'EURGBP': 2,
    'EURUSD': 1.5,
    'USDCHF': 2,
    'USDJPY': 1.5,
    'EURPLN': 30
}

# Cost is expressed in percentages
COST_CRYPTO = {
    'BTCUSD': 0.0025,
    'ETHUSD': 0.0025,
    'BTCUSDT': 0.0025,
    'ETHUSDT': 0.0025,
    'MATICUSDT': 0.0025,
    'SOLUSDT': 0.0025
}

try:
    cost = COST_FOREX[currency] # transactional costs
except KeyError:
    cost = COST_CRYPTO[currency]
except KeyError:
    cost = 0

# XTB AUTHENTICATION - OPTIONAL
USER_ID = ""
PASSWORD = ""

# Ending dates for training, validation and test. Remember that xtb offers much shorther periods, so
# if using training from xtb folder adjust those dates. Otherwise, program will fail at
# train, validation split
train_end = datetime.datetime(2019, 12, 31, 0, 0, 0)
val_end = datetime.datetime(2020, 12, 31, 0, 0, 0)
test_end = datetime.datetime(2021, 12, 31, 0, 0, 0)
# train_end = datetime.datetime(2021, 12, 31, 0, 0, 0)
# val_end = datetime.datetime(2022, 7, 1, 0, 0, 0)
# test_end = datetime.datetime(2023, 1, 31, 0, 0, 0)

# flag if for multi prediction period we should drop observations in between to avoid leaking target. Drawback is losing many obs
time_step = False

# model hyperparameters
seq_len = 96
batch = 64
patience = 15
epochs = 100
nb_classes = 2
steps_ahead = 1

# configure directories

if RUN_TYPE == 'crypto':
    package_directory = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(package_directory, 'data', 'raw', 'crypto')
elif RUN_TYPE == 'forex':
    package_directory = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(package_directory, 'data', 'raw', 'hist_data')
    raw_data_path_xtb = os.path.join(package_directory, 'data', 'raw', 'xtb')
    raw_data_path_test = os.path.join(package_directory, 'data', 'test')
else:
    pass
# common directories
processed_data_path = os.path.join(package_directory, 'data', 'processed')
model_path = os.path.join(package_directory, 'models',f'{model}_{currency}_{freq}.hdf5')
model_path_final = os.path.join(package_directory, 'models',f'{model}_{currency}_{freq}.h5')