import os
import datetime

currency = 'USDCHF'
# leave empty if training from scratch
load_model = ''
freq = '60min'
input_dim = '2d'  # 2d or 1d
model = 'inception_2d'
train_end = datetime.datetime(2021, 8, 1, 0, 0, 0)
val_end = datetime.datetime(2021, 10, 1, 0, 0, 0)
test_end = datetime.datetime(2021, 11, 30, 0, 0, 0)
seq_len = 96
batch = 128
patience = 8
epochs = 100
nb_classes = 2
steps_ahead = 0  # 0 for 1 step ahead

# configure directories
package_directory = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(package_directory, 'data', 'raw')
raw_data_path_test = os.path.join(package_directory, 'data', 'test')
processed_data_path = os.path.join(package_directory, 'data', 'processed')
model_path = os.path.join(package_directory, 'models',
                               '{}_{}_{}.hdf5'.format(model, currency, freq))

model_path_final = os.path.join(package_directory, 'models',
                               '{}_{}_{}.h5'.format(model, currency, freq))


