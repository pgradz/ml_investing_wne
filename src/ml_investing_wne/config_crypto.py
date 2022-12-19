import os
import datetime

exchange = 'Bitstamp'
currency = 'ETHUSD'

freq = '60min'
input_dim = '1d'  # 2d or 1d
# has to be defined inside tf_models folder
model = 'resnet'
seed = 12345

train_end = datetime.datetime(2021, 1, 31, 0, 0, 0)
val_end = datetime.datetime(2021, 3, 1, 0, 0, 0)
test_end = datetime.datetime(2021, 3, 31, 0, 0, 0)

# model hyperparameters
seq_len = 24
batch = 64
patience = 15
epochs = 100
nb_classes = 2
steps_ahead = 1

package_directory = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(package_directory, 'data', 'raw', 'crypto')