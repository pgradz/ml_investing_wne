import os
import datetime

# hide GPU from tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# # suppress tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

run_type = 'crypto' # forex or crypto
run_subtype = 'time_aggregated' #'triple_barrier_time_aggregated','time_aggregated', 'volume_bars', dollar_bars, 'cumsum', cumsum_triple_barrier', 'volume_bars_triple_barrier', dollar_bars_triple_barrier, range_bar, range_bar_triple_barrier
provider = 'Binance' # hist_data, Bitstamp, Binance
currency = 'ADAUSDT'

# model parameters
input_dim = '1d'  # 2d or 1d
# has to be defined inside tf_models folder
model = 'keras_tuner_CNN_LSTM' #'keras_tuner_transformer_positional_encoding', resnet_lstm_regularized, transformer_learnable_encoding, lstm, keras_tuner_CNN_LSTM, keras_tuner_LSTM

# time bars
freq = '1min'
# volume for volume bars
volume = 20000
value = 300000000

# Tripple barrier params
t_final=24
fixed_barrier=0.05

# cusum andrange bars params
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
    'LTCUSDT': 0.001,
    'DOGEUSDT': 0.001,
    'ADAUSDT': 0.001,
    'LINKUSDT': 0.001
}

try:
    cost = COST_FOREX[currency] # transactional costs
except KeyError:
    cost = COST_CRYPTO[currency]
except KeyError:
    cost = 0

# model hyperparameters
seq_len = 96
batch_size = 128
patience = 3
epochs = 100
nb_classes = 2
steps_ahead = 1

# configure directories
if run_type == 'crypto':
    package_directory = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(package_directory, 'data', 'raw', 'crypto')
elif run_type == 'forex':
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