import os
import datetime

currency = 'EURPLN'
# leave empty if training from scratch
load_model = 'USDJPY'
freq = '60min'
input_dim = '2d'  # 2d or 1d
model = 'deepLOB'
train_end = datetime.datetime(2021, 9, 1, 0, 0, 0)
val_end = datetime.datetime(2021, 10, 1, 0, 0, 0)
test_end = datetime.datetime(2021, 10, 30, 0, 0, 0)
seq_len = 96
batch = 128
patience = 5
epochs = 100
nb_classes = 2

# configure directories
package_directory = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(package_directory, 'data', 'raw')
raw_data_path_test = os.path.join(package_directory, 'data', 'test')
processed_data_path = os.path.join(package_directory, 'data', 'processed')
model_path = os.path.join(package_directory, 'models',
                               '{}_{}_{}.hdf5'.format(model, currency, freq))

model_path_final = os.path.join(package_directory, 'models',
                               'resnet_{}_{}.h5'.format(currency, freq))


