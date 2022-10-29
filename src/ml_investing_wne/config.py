import os
import datetime

currency = 'EURCHF'
# leave empty if training from scratch
load_model = ''
freq = '720min'
input_dim = '1d'  # 2d or 1d
model = 'transformer_learnable_encoding'

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

userId = ""
password = ""

# for xtb training:
# train_end = datetime.datetime(2020, 12, 31, 0, 0, 0)
# val_end = datetime.datetime(2021, 12, 31, 0, 0, 0)
# test_end = datetime.datetime(2022, 9, 23, 0, 0, 0)
# train_end = datetime.datetime(2020, 12, 30, 0, 0, 0)
# val_end = datetime.datetime(2021, 7, 1, 0, 0, 0)
# test_end = datetime.datetime(2021, 12, 30, 0, 0, 0)
# for article:
train_end = datetime.datetime(2019, 12, 31, 0, 0, 0)
val_end = datetime.datetime(2020, 12, 31, 0, 0, 0)
test_end = datetime.datetime(2021, 12, 31, 0, 0, 0)
# test_end = datetime.datetime(2022, 8, 31, 0, 0, 0)

seq_len = 96
batch = 128
patience = 15
epochs = 100
nb_classes = 2
steps_ahead = 1

# configure directories
package_directory = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(package_directory, 'data', 'raw', 'hist_data')
raw_data_path_xtb = os.path.join(package_directory, 'data', 'raw', 'xtb')
raw_data_path_test = os.path.join(package_directory, 'data', 'test')
processed_data_path = os.path.join(package_directory, 'data', 'processed')
model_path = os.path.join(package_directory, 'models',
                               '{}_{}_{}.hdf5'.format(model, currency, freq))

model_path_final = os.path.join(package_directory, 'models',
                               '{}_{}_{}.h5'.format(model, currency, freq))


