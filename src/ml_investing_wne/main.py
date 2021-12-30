import datetime
import os
import logging
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import mlflow.keras
import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes
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
mlflow.tensorflow.autolog()

df = prepare_processed_dataset()
# df.drop(columns=['ask_open', 'ask_high', 'ask_min', 'ask_close', 'spread_open', 'spread_high', 'spread_min', 'currency']
#         , axis=1, inplace=True)
df.drop(columns=['ask_open', 'ask_high', 'ask_min', 'ask_close',  'currency']
        , axis=1, inplace=True)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len)

mlflow.set_experiment(experiment_name=config.model + '_' + str(config.nb_classes))
early_stop = EarlyStopping(monitor='val_accuracy', patience=config.patience)
model_checkpoint = ModelCheckpoint(filepath=config.model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]

# continue training or start new model
if len(config.load_model) > 1:
    model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_{}_{}.hdf5'.format(config.model, config.load_model, config.freq)))
else:
    model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=config.nb_classes)

history = model.fit(X, y_cat, batch_size=config.batch, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)

# export model architecture and header of training dataset so they can be attached to mlflow
plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot.png'), show_shapes=True,
           show_layer_names=True)
train.loc[:1].to_csv(os.path.join(config.package_directory, 'models', 'features.csv'))
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'model_plot.png'))
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'features.csv'))
model.summary()
model.save(config.model_path_final)
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
logger.info('Test accuracy : {}'.format(test_acc))
logger.info('Test loss : {}'.format(test_loss))
mlflow.log_metric("test_acc", test_acc)
mlflow.log_metric("test_loss", test_loss)
mlflow.log_metric("test_loss", test_loss)
mlflow.set_tag('currency', config.currency)

y_pred = model.predict(X_test)
confusion_matrix_plot(y_pred, y_test)
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'confusion_matrix_{}_{}_{}.png'.
                                 format(config.model, config.currency, config.nb_classes)))

pips = 1
df['cost'] = (pips/10000)/df['close']
portfolio_result = compute_profitability_classes(df, y_pred, datetime.datetime(2021, 10, 7, 0, 0, 0),
                                                 datetime.datetime(2021, 11, 29, 23, 0, 0))
mlflow.log_metric("portfolio_result", portfolio_result)

mlflow.log_artifact(os.path.join(config.package_directory, 'models',
                                                        'portfolio_evolution_{}_{}_{}.png'.
                                                        format(config.model, config.currency, config.nb_classes)))
mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}.csv'.
                                 format(config.model, config.currency, config.nb_classes)))

# mlflow ui --backend-store-uri /Users/i0495036/Documents/sandbox/ml_investing_wne/mlruns


# print(model.summary())
#
# manual checks
prediction = df.copy()
prediction.reset_index(inplace=True)
df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
# new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
prediction = df.loc[(df.datetime >= datetime.datetime(2021, 10, 7, 0, 0, 0)) & (df.datetime <= datetime.datetime(2021, 11, 29, 23, 0, 0))]
prediction['trade'] = y_pred.argmax(axis=1)
prediction.reset_index(inplace=True)
prediction['y_prob'] = y_pred[:, 1]

correct_predictions = prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5) ].shape[0]
correct_predictions =  correct_predictions + prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < 0.5)].shape[0]
correct_predictions / prediction.shape[0]

prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.7)].shape[0]/ prediction.loc[(prediction['y_prob'] > 0.7)].shape[0]

prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5) & (prediction['y_prob'] < 0.53)].shape[0] / prediction.loc[ (prediction['y_prob'] > 0.5) & (prediction['y_prob'] < 0.53)].shape[0]
import pandas as pd
prediction['correct_prediction'] = 0
prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5), 'correct_prediction' ] = 1
prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < 0.5), 'correct_prediction' ] = 1
prediction['correct_prediction'].mean()
prediction['probability_binned'] = pd.cut(prediction['y_prob'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
prediction.groupby('probability_binned')['correct_prediction'].mean()
prediction.groupby('probability_binned')['correct_prediction'].count()
