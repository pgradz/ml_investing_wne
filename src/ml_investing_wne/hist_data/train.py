import os
import pandas as pd
import logging
import datetime
import joblib
import mlflow.keras
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes, check_hours
from ml_investing_wne.hist_data.helper import get_hist_data
import importlib
import matplotlib.pyplot as plt



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

mlflow.tensorflow.autolog()
# sc_x = joblib.load(os.path.join(config.package_directory, 'models',
#                                    'sc_x_{}_{}.save'.format(config.currency, config.freq)))
# model = load_model(os.path.join(config.package_directory, 'models',
#                                     '{}_{}_{}.hdf5'.format(config.model, config.currency, config.freq)))

df = get_hist_data(currency=config.currency)
df = prepare_processed_dataset(df=df)

# df = df[np.random.default_rng(seed=42).permutation(df.columns.values)]

X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len)

mlflow.set_experiment(experiment_name='hist_data' + '_' + config.model + '_' + str(config.nb_classes) + '_' + \
                                      config.freq +'_' + str(config.steps_ahead) + '_' + str(config.seq_len))
early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
model_path_final = os.path.join(config.package_directory, 'models',
                               '{}_{}_{}_{}_{}.h5'.format(config.model, 'hist_data', config.currency, config.freq, config.steps_ahead))
model_checkpoint = ModelCheckpoint(filepath=model_path_final, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]
# model = load_model('/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/models/resnet_lstm_2_hist_data_EURUSD_480min_1.h5')
# config.load_model = ''
# continue training or start new model
if len(config.load_model) > 1:
    # model = load_model(os.path.join(config.package_directory, 'models',
    #                                 '{}_{}_{}_{}.h5'.format(config.model, 'hist_data', config.load_model, config.freq)))
    model = load_model(os.path.join(config.package_directory, 'models', 'production',
                                    '{}_hist_data_{}_{}_{}_{}'.format(config.model, config.load_model, config.freq, config.steps_ahead,config.seq_len )))
else:
    model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=config.nb_classes)

# transformer
model = build_model(input_shape=(96, 40), head_size=256, num_heads=4, ff_dim=32,
                     num_transformer_blocks=2, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

history = model.fit(X, y_cat, batch_size=64, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)

model.save(os.path.join(config.package_directory, 'models', 'production',
                                '{}_{}_{}_{}_{}_{}'.format(config.model, 'hist_data', config.currency, config.freq, str(config.steps_ahead), config.seq_len)))

# model2.evaluate(X_val, y_val_cat)
# test_loss, test_acc = model2.evaluate(X_test, y_test_cat)

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
mlflow.log_metric('roc_auc', roc_auc)
mlflow.log_metric('f1', f1)

if 'JPY' in config.currency:
    df['cost'] = (config.pips/100)/df['close']
else:
    df['cost'] = (config.pips/10000)/df['close']


start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'first_sequence_ends_{}_{}_{}.save'.format('test', config.currency, config.freq)))
end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'last_sequence_ends_{}_{}_{}.save'.format('test', config.currency, config.freq)))
lower_bounds =[0.1,0.15,0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5]
upper_bounds = [1 - lower for lower in lower_bounds]

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound)
    mlflow.log_metric("portfolio_result_{}_{}".format(lower_bound, upper_bound), portfolio_result)
    mlflow.log_metric("hit_ratio_{}_{}".format(lower_bound, upper_bound), hit_ratio)
    mlflow.log_metric("time_active_{}_{}".format(lower_bound, upper_bound), time_active)
    mlflow.log_artifact(os.path.join(config.package_directory, 'models',
                                     'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                                     format(config.model, config.currency, config.nb_classes,
                                            lower_bound, upper_bound)))

mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}_{}.csv'.
                                 format(config.model, config.currency, config.nb_classes, config.steps_ahead)))

predicton = check_hours(df, y_pred, start_date, end_date, lower_bound=0.5, upper_bound=0.5)
predicton = check_hours(df, y_pred, start_date, end_date, lower_bound=0.45, upper_bound=0.55)
predicton = check_hours(df, y_pred, start_date, end_date, lower_bound=0.4, upper_bound=0.6)

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound,
                                                                             time_waw_list=[datetime.time(22,0,0)
                                                                                ])

predicton['pips_difference'] = (predicton['close'].shift(-config.steps_ahead) - predicton['close'])*10000
predicton['pips_difference'] = (predicton['high'].shift(-config.steps_ahead) - predicton['close'])*10000
# for JPY
#predicton['pips_difference'] = (predicton['close'].shift(-1) - predicton['close'])*100

t2h = [
datetime.time(2,0,0),
datetime.time(4,0,0),
datetime.time(6,0,0),
datetime.time(8,0,0),
datetime.time(10,0,0),
datetime.time(12,0,0),
datetime.time(14,0,0),
datetime.time(16,0,0),
datetime.time(18,0,0),
datetime.time(20,0,0),
datetime.time(22,0,0),
datetime.time(0,0,0),
]

predicton.boxplot(column = 'pips_difference', by = 'hour_waw', figsize=(30,10))
predicton.loc[predicton['hour_waw']==datetime.time(22,0,0),'pips_difference'].describe(percentiles=[0.25, 0.33, 0.4, 0.45, 0.5, 0.6, 0.66, 0.75])
predicton.loc[(predicton['hour_waw']==datetime.time(23,0,0)) & (predicton['pips_difference']>0)].shape[0]/predicton.loc[predicton['hour_waw']==datetime.time(21,0,0)].shape[0]

#predicton.loc[predicton['hour_waw'].isin(t2h)].boxplot(column = 'pips_difference', by = 'hour_waw', figsize=(30,10))
plt.title('Difference in price in pips between consecutive periods for {}'.format(config.currency))
plt.xticks(rotation=90)
plt.savefig(os.path.join(config.package_directory, 'models', 'Price_difference_{}_{}_{}.png'.
                         format(config.model, config.currency, config.freq)))

predicton['sign'] = [1 if y > 0 else 0 for y in predicton['y_pred']]
predicton.groupby('hour_waw')['sign'].mean()
#predicton.loc[predicton['hour_waw'].isin(t2h)].groupby('hour_waw')['sign'].mean()

# full dataset
df['datetime_waw'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
    'Europe/Warsaw').dt.tz_localize(None)
df['hour_waw'] = df['datetime_waw'].dt.time
df['pips_difference'] = (df['close'].shift(-1) - df['close'])*10000
df['pips_difference'] = (df['high'].shift(-1) - df['close'])*10000

df.boxplot(column = 'pips_difference', by = 'hour_waw', figsize=(30,10))
df.loc[(df['pips_difference']<200) & (df['pips_difference']>-200)].boxplot(column = 'pips_difference', by = 'hour_waw', figsize=(30,10))
plt.title('Difference in price in pips between consecutive periods for {} full period'.format(config.currency))
plt.xticks(rotation=90)
plt.savefig(os.path.join(config.package_directory, 'models', 'Price_difference_full_period_{}_{}_{}.png'.
                         format(config.model, config.currency, config.freq)))


df.loc[df['hour_waw']==datetime.time(21,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
df.loc[df['hour_waw']==datetime.time(22,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
df.loc[df['hour_waw']==datetime.time(23,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
predicton.loc[predicton['hour_waw']==datetime.time(20,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
predicton.loc[predicton['hour_waw']==datetime.time(21,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
predicton.loc[predicton['hour_waw']==datetime.time(22,0,0)]['pips_difference'].describe(percentiles=[0.1,0.25,0.33,0.4,0.5,0.6,0.66,0.75,0.8,0.9])
predicton.loc[predicton['hour_waw']==datetime.time(22,0,0)]['pips_difference'].mean()
df.loc[df['hour_waw']==datetime.time(22,0,0)]['pips_difference'].mean()


df_top_n = df_melted_index.sort_values('value',ascending = False).groupby('cluster_name')\
    .head(n_features).copy()

df_bottom_n = df_melted_index.sort_values('value',ascending = True).groupby('cluster_name').\
    head(n_features).copy()


