import os
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import mlflow.keras
import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.train_test_val_split import train_test_val_split
import importlib

# load model dynamically
build_model = getattr(importlib.import_module('ml_investing_wne.cnn.{}'.format(config.model)), 'build_model')

# logger settings
logger = logging.getLogger()
# You can set a different logging level for each logging handler but it seems you will have to set the
# logger's level to the "lowest".
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
file_h = logging.FileHandler(os.path.join(config.package_directory, 'logs', 'app.log'))
stream_h.setLevel(logging.INFO)
file_h.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
stream_h.setFormatter(formatter)
file_h.setFormatter(formatter)
logger.addHandler(stream_h)
logger.addHandler(file_h)

# autlog ends run after keras.fit, hence all logging later will be added to another run. This is inconvenient, but
# the alternative is to manually create run and replicate functionalities of autlog
mlflow.keras.autolog()

df = prepare_processed_dataset()
df.drop(columns=['ask_open', 'ask_high', 'ask_min', 'ask_close', 'spread_open', 'spread_high', 'spread_min', 'currency']
        , axis=1, inplace=True)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat , train = train_test_val_split(df, config.seq_len)



mlflow.set_experiment(experiment_name=config.model)
early_stop = EarlyStopping(monitor='val_loss', patience=config.patience)
model_checkpoint = ModelCheckpoint(filepath=config.model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs','keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]
model = load_model(os.path.join(config.package_directory, 'models',
                               'deepLOB_{}_{}.hdf5'.format('USDCHF', config.freq)))
# model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=2)

history = model.fit(X, y_cat, batch_size=config.batch, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)

# export model architecture and header of training dataset so they can be attached to mlflow
plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot.png'), show_shapes=True,
           show_layer_names=True)
train.loc[:1].to_csv(os.path.join(config.package_directory, 'models', 'features.csv'))
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'model_plot.png'))
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'features.csv'))

model.save(config.model_path_final)
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
logger.info('Test accuracy : {}'.format(test_acc))
logger.info('Test loss : {}'.format(test_loss))
mlflow.log_metric("test_acc", test_acc)
mlflow.log_metric("test_loss", test_loss)
