import os
import datetime


RUN_TYPE = 'crypto' # forex or crypto
provider = 'Bitstamp' # hist_data or Bitstamp

currency = 'BTCUSD'
# leave empty if training from scratch, for transfer learning specify currency to be used as a base
load_model = ''
freq = '5min'
input_dim = '1d'  # 2d or 1d
# has to be defined inside tf_models folder
model = 'resnet_lstm_regularized'
seed = 12345

# Cost is expressed in pips
COST = {
    'EURCHF': 2,
    'EURGBP': 2,
    'EURUSD': 1.5,
    'USDCHF': 2,
    'USDJPY': 1.5,
    'EURPLN': 30
}
try:
    pips = COST[currency] # transactional costs
except KeyError:
    pips = 0

# XTB AUTHENTICATION - OPTIONAL
USER_ID = ""
PASSWORD = ""

# Ending dates for training, validation and test. Remember that xtb offers much shorther periods, so
# if using training from xtb folder adjust those dates. Otherwise, program will fail at
# train, validation split
# train_end = datetime.datetime(2019, 12, 31, 0, 0, 0)
# val_end = datetime.datetime(2020, 12, 31, 0, 0, 0)
# test_end = datetime.datetime(2021, 12, 31, 0, 0, 0)
train_end = datetime.datetime(2021, 3, 31, 0, 0, 0)
val_end = datetime.datetime(2021, 7, 1, 0, 0, 0)
test_end = datetime.datetime(2021, 11, 26, 0, 0, 0)

# model hyperparameters
seq_len = 48
batch = 64
patience = 20
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