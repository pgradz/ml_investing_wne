import datetime
import os
import random
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import tensorflow as tf
import mlflow.keras
import importlib
import joblib
from sklearn.metrics import roc_auc_score, f1_score

import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes
from ml_investing_wne.utils import get_logger

seed = 12345
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
                   
logger = get_logger()

# load model dynamically
build_model = getattr(importlib.import_module('ml_investing_wne.cnn.{}'.format(config.model)),
                      'build_model')

# autlog ends run after keras.fit, hence all logging later will be added to another run. This is
# inconvenient, but the alternative is to manually create run and replicate functionalities of autlog
mlflow.tensorflow.autolog()

df = get_hist_data(currency=config.currency)
df = prepare_processed_dataset(df=df)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df)

mlflow.set_experiment(experiment_name='hist_data' + '_' + config.model + '_' +
                                      str(config.nb_classes) + '_' +
                                      config.freq + '_' + str(config.steps_ahead) + '_' +
                                      str(config.seq_len))
early_stop = EarlyStopping(monitor='val_accuracy', patience=config.patience, restore_best_weights=True)
model_path_final = os.path.join(config.package_directory, 'models',
                                '{}_{}_{}_{}_{}.h5'.format(config.model, 'hist_data',
                                                           config.currency, config.freq,
                                                           config.steps_ahead))
model_checkpoint = ModelCheckpoint(filepath=model_path_final, monitor='val_accuracy', verbose=1,
                                   save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True,
                       separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]


# model = load_model(os.path.join(config.package_directory, 'models',
#                                     '{}_hist_data_{}_{}_{}.h5'.format(config.model,
#                                                                       config.load_model,
#                                                                       config.freq,
#                                                                       config.steps_ahead)))
# continue training or start new model
# if len(config.load_model) > 1:
#     # models have to be moved to production folder in order to be used
#     model = load_model(os.path.join(config.package_directory, 'models', 'production',
#                                     '{}_hist_data_{}_{}_{}_{}'.format(config.model,
#                                                                       config.load_model,
#                                                                       config.freq,
#                                                                       config.steps_ahead,
#                                                                       config.seq_len)))
# else:
#     model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=config.nb_classes)

model = build_model(input_shape=(config.seq_len, 40), head_size=64, num_heads=4, ff_dim=64,
                    num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.25, dropout=0.25)
# model = build_model(input_shape=(96, 40), head_size=64, num_heads=4, ff_dim=64, embedding_size=64,
#                    num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.25, dropout=0.25)
history = model.fit(X, y_cat, batch_size=64, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)

model.save(os.path.join(config.package_directory, 'models', 'production',
                        '{}_{}_{}_{}_{}_{}'.format(config.model, 'hist_data',
                                                   config.currency, config.freq,
                                                   str(config.steps_ahead),
                                                   config.seq_len)))
model.evaluate(X_val, y_val_cat)
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
logger.info('Test accuracy : {}'.format(test_acc))
logger.info('Test loss : {}'.format(test_loss))
mlflow.log_metric("test_acc", test_acc)
mlflow.log_metric("test_loss", test_loss)
mlflow.log_metric("test_loss", test_loss)
mlflow.set_tag('currency', config.currency)
mlflow.set_tag('frequency', config.freq)
mlflow.set_tag('steps_ahead', config.steps_ahead)
mlflow.log_metric('y_distribution', y.mean())
mlflow.log_metric('y_val_distribution', y_val.mean())
mlflow.log_metric('y_test_distribution', y_test.mean())
mlflow.log_metric('cost', config.pips)
mlflow.log_metric('seq_len', config.seq_len)

y_pred = model.predict(X_test)
y_pred_class = y_pred.argmax(axis=-1)

roc_auc = roc_auc_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
logger.info('roc_auc : {}'.format(roc_auc))
logger.info('f1 : {}'.format(f1))
mlflow.log_metric('roc_auc', roc_auc)
mlflow.log_metric('f1', f1)

if 'JPY' in config.currency:
    df['cost'] = (config.pips / 100) / df['close']
else:
    df['cost'] = (config.pips / 10000) / df['close']

start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                      'first_sequence_ends_{}_{}_{}.save'.format('test',
                                                                                 config.currency,
                                                                                 config.freq)))
end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                    'last_sequence_ends_{}_{}_{}.save'.format('test',
                                                                              config.currency,
                                                                              config.freq)))
lower_bounds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
upper_bounds = [1 - lower for lower in lower_bounds]

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date,
                                                                             end_date, lower_bound,
                                                                             upper_bound)
    mlflow.log_metric("portfolio_result_{}_{}".format(lower_bound, upper_bound), portfolio_result)
    mlflow.log_metric("hit_ratio_{}_{}".format(lower_bound, upper_bound), hit_ratio)
    mlflow.log_metric("time_active_{}_{}".format(lower_bound, upper_bound), time_active)
    mlflow.log_artifact(os.path.join(config.package_directory, 'models',
                                     'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                                     format(config.model, config.currency, config.nb_classes,
                                            lower_bound, upper_bound)))

# mlflow.log_artifact(
#     os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}.csv'.
#                  format(config.model, config.currency, config.nb_classes)))

# mlflow ui --backend-store-uri /Users/i0495036/Documents/sandbox/ml_investing_wne/mlruns
#  ps -A | grep gunicorn
# mlflow ui --port=5001 --backend-store-uri /Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/mlruns
# print(model.summary())
#
# # manual checks
# prediction = df.copy()
# prediction.reset_index(inplace=True)
# df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
# # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
# prediction = df.loc[(df.datetime >= datetime.datetime(2021, 10, 7, 0, 0, 0)) & (df.datetime <= datetime.datetime(2021, 11, 29, 23, 0, 0))]
# prediction['trade'] = y_pred.argmax(axis=1)
# prediction.reset_index(inplace=True)
# prediction['y_prob'] = y_pred[:, 1]
#
# correct_predictions = prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5) ].shape[0]
# correct_predictions =  correct_predictions + prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < 0.5)].shape[0]
# correct_predictions / prediction.shape[0]
#
# prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.7)].shape[0]/ prediction.loc[(prediction['y_prob'] > 0.7)].shape[0]
#
# prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5) & (prediction['y_prob'] < 0.53)].shape[0] / prediction.loc[ (prediction['y_prob'] > 0.5) & (prediction['y_prob'] < 0.53)].shape[0]
# import pandas as pd
# prediction['correct_prediction'] = 0
# prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5), 'correct_prediction' ] = 1
# prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < 0.5), 'correct_prediction' ] = 1
# prediction['correct_prediction'].mean()
# prediction['probability_binned'] = pd.cut(prediction['y_prob'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# prediction.groupby('probability_binned')['correct_prediction'].mean()
# prediction.groupby('probability_binned')['correct_prediction'].count()
