import os
import datetime

# hide GPU from tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# # suppress tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TYPE = 'crypto' # forex or crypto
RUN_SUBTYPE = 'volume_bars_triple_barrier' #'triple_barrier_time_aggregated','time_aggregated', 'volume_bars', dollar_bars, 'cumsum', cumsum_triple_barrier', 'volume_bars_triple_barrier', dollar_bars_triple_barrier, range_bar, range_bar_triple_barrier
provider = 'Binance' # hist_data, Bitstamp, Binance
currency = 'BTCUSDT'

# model parameters
input_dim = '1d'  # 2d or 1d
# has to be defined inside tf_models folder
model = 'keras_tuner_transformer_learnable_encoding' #'keras_tuner_transformer_positional_encoding', resnet_lstm_regularized, transformer_learnable_encoding, lstm, keras_tuner_CNN_LSTM, keras_tuner_LSTM
seed = 12345
# leave empty if training from scratch, for transfer learning specify currency to be used as a base
load_model = ''

# time bars
freq = '720min'

# volume for volume bars
volume = 5000
value = 150000000


# Tripple barrier params
t_final=24
fixed_barrier=0.025

# cumsum params
cumsum_threshold = 0.03


# if we want to skip consequtive sequences, it is configured by seq_stride. If seq_stride = seq_len then there is 0 overlap at expense of many observations dropped
seq_stride = 1

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
    'BTCUSD': 0.001,
    'ETHUSD': 0.001,
    'BTCUSDT': 0.001,
    'ETHUSDT': 0.001,
    'MATICUSDT': 0.001,
    'SOLUSDT': 0.001,
    'ETHBTC': 0.001,
    'LTCUSDT': 0.001
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
# train_end = datetime.datetime(2021, 6, 1, 0, 0, 0)
# val_end = datetime.datetime(2021, 9, 1, 0, 0, 0)
# test_end = datetime.datetime(2021, 12, 1, 0, 0, 0)
# train_end = datetime.datetime(2022, 1, 1, 0, 0, 0)
# val_end = datetime.datetime(2022, 7, 1, 0, 0, 0)
# test_end = datetime.datetime(2023, 1, 1, 0, 0, 0)
train_end = datetime.datetime(2022, 1, 1, 0, 0, 0)
val_end = datetime.datetime(2022, 4, 1, 0, 0, 0)
test_end = datetime.datetime(2022, 7, 1, 0, 0, 0)
# train_end = datetime.datetime(2022, 7, 1, 0, 0, 0)
# val_end = datetime.datetime(2023, 1, 1, 0, 0, 0)
# test_end = datetime.datetime(2023, 7, 1, 0, 0, 0)

# model hyperparameters
seq_len = 96
batch = 128
patience = 3
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