import itertools
import logging
import os
import re

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import ml_investing_wne.config as config

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def get_ml_flow_experiment_name():
    """creates mlflow experiment name based on config parameters.

    Returns:
        str: mlflow experiment name
    """

    ml_flow_experiment_name = (f"{config.provider}_{config.model}_{str(config.nb_classes)}"
                                f"{config.freq}_{str(config.steps_ahead)}_{str(config.seq_len)}")
    return ml_flow_experiment_name


def get_training_model_path():
    """_summary_

    Returns:
        _type_: _description_
    """
    model_name = f"{config.model}_{config.provider}_{config.currency}_{config.freq}_{config.steps_ahead}.h5"
    model_path_training = os.path.join(config.package_directory, 'models', model_name)

    return model_path_training

def get_xtb_training_model_path():
    """_summary_

    Returns:
        _type_: _description_
    """
    model_name = f"{config.model}_hist_data_{config.currency}_{config.freq}_{config.steps_ahead}.h5"
    model_path_training = os.path.join(config.package_directory, 'models', model_name)

    return model_path_training

def get_final_model_path():

    model_name = (f"{config.model}_{config.provider}_{config.currency}_{config.freq}_"
                    f"{str(config.steps_ahead)}_{config.seq_len}")
    model_path_final = os.path.join(config.package_directory, 'models', 'production', model_name)
    return model_path_final

def get_callbacks():
    
    if config.provider == 'xtb':
        training_model_path = get_xtb_training_model_path()
    else:
        training_model_path = get_training_model_path()

    early_stop = EarlyStopping(monitor='val_accuracy', patience=config.patience, 
                restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=training_model_path, monitor='val_accuracy',
                        verbose=1, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), 
                            append=True, separator=';')
    callbacks = [early_stop, model_checkpoint, csv_logger]

    return callbacks

def get_scaler():
    
    sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                   f'sc_x_{config.currency}_{config.freq}.save'))
    return sc_x

def confusion_matrix_plot(y_pred, y_test):
    
    y_pred.argmax(axis=1)
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    plt.figure(1)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix (precision) for {}'.format(config.currency))
    plt.colorbar()
    tick_marks = np.arange(config.nb_classes)
    plt.xticks(tick_marks, range(config.nb_classes), rotation=45)
    plt.yticks(tick_marks, range(config.nb_classes))

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, round(matrix[i, j] / matrix[:, j].sum(), 2),
                 horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(config.package_directory, 'models', 'confusion_matrix_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes)))


def evaluate_model(model, df, X_test, y_test_cat, y, y_val, y_test):

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    logger.info('Test accuracy : %.4f', test_acc)
    logger.info('Test loss : %.4f', test_loss)
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
    logger.info('roc_auc : %.4f',roc_auc)
    logger.info('f1 : %.4f', f1)
    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_metric('f1', f1)

    df = add_cost(df)
    start_date, end_date = load_test_dates()

    lower_bounds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    upper_bounds = [1 - lower for lower in lower_bounds]

    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, 
                                                                                start_date,
                                                                                end_date, 
                                                                                lower_bound,
                                                                                upper_bound)
        mlflow.log_metric(f"portfolio_result_{lower_bound}_{upper_bound}", portfolio_result)
        mlflow.log_metric(f"hit_ratio_{lower_bound}_{upper_bound}", hit_ratio)
        mlflow.log_metric(f"time_active_{lower_bound}_{upper_bound}",time_active)
        name = f'''portfolio_evolution_{config.model}_{config.currency}_{config.nb_classes}_
                {lower_bound}_{upper_bound}.png'''
        mlflow.log_artifact(os.path.join(config.package_directory, 'models', name))

def load_test_dates():

    name = f'test_{config.currency}_{config.freq}.save'

    start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                        f'first_sequence_ends_{name}'))
    end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                        f'last_sequence_ends_{name}'))

    return start_date, end_date

def add_cost(df):
    if 'JPY' in config.currency:
        df['cost'] = (config.pips / 100) / df['close']
    else:
        df['cost'] = (config.pips / 10000) / df['close']
    return df


# #
# upper_bound = 0.5
# lower_bound = 0.5
# date_start = start_date
# date_end = end_date

def compute_profitability_classes(df, y_pred, date_start, date_end, lower_bound, upper_bound,
                                  hours_exclude=None):
    
    # recreate target as continous variable
    df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close'] - 1
    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)]
    if config.provider == 'hist_data':
        prediction['datetime_local'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
            'Europe/London').dt.tz_localize(None)
    else:
        prediction['datetime_local'] = prediction['datetime']

    prediction['hour_local'] = prediction['datetime_local'].dt.time
    # TODO: here triple barriers fails
    prediction['prediction'] = y_pred[:, 1]
    conditions = [
        (prediction['prediction'] <= lower_bound),
        (prediction['prediction'] > lower_bound) & (prediction['prediction'] <= upper_bound),
        (prediction['prediction'] > upper_bound)
    ]
    values = [0, 0.5, 1]
    prediction['trade'] = np.select(conditions, values)
    if hours_exclude:
        prediction.loc[prediction['hour_local'].isin(hours_exclude), 'trade'] = 0.5
    prediction.reset_index(inplace=True)
    # drop last row for which we don't have a label - this works only for one step ahead prediction
    prediction.drop(prediction.tail(1).index, inplace=True)

    # INITIALIZE PORTFOLIO
    budget = 100
    transaction = None
    i = 0

    # ITERATE OVER PREDICTIONS
    # cost is added once as it represents spread
    while i < prediction.shape[0]:
    
        if prediction.loc[i, 'trade'] == 1:
            # add transaction cost if position changes
            if transaction != 'buy':
                budget = budget * (1 - prediction.loc[i, 'cost'])
            transaction = 'buy'
            budget = budget + budget * prediction.loc[i, 'y_pred']
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + config.steps_ahead
        elif prediction.loc[i, 'trade'] == 0:
            # add transaction cost if position changes
            if transaction != 'sell':
                budget = budget * (1 - prediction.loc[i, 'cost'])
            transaction = 'sell'
            budget = budget + budget * (-prediction.loc[i, 'y_pred'])
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + config.steps_ahead
        elif prediction.loc[i, 'trade'] == 0.5:
            if transaction in ['buy', 'sell']:
                # budget = budget * (1 - prediction.loc[i, 'cost']) # spread is included once in transaction costs
                transaction = None
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + 1

    # SUMMARIZE RESULTS
    hits = prediction.loc[((prediction['transaction'] == 'buy') & (prediction['y_pred'] > 0)) |
                          ((prediction['transaction'] == 'sell') & (prediction['y_pred'] < 0))].shape[0]
    transactions = prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0]
    try:
        hits_ratio = hits / transactions
    except ZeroDivisionError:
        hits_ratio = 0
    share_of_time_active = round(prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0] * \
                                 config.steps_ahead / prediction.shape[0], 2)

    logger.info('''share_of_time_active for bounds %.2f-%.2f is %.2f and hit ratio is %.4f''',
                lower_bound, upper_bound, share_of_time_active, hits_ratio)
    logger.info('Portfolio result:  %.2f', budget)

    plot_portfolio(prediction, lower_bound, upper_bound)
    
    return budget, hits_ratio, share_of_time_active

def plot_portfolio(prediction, lower_bound, upper_bound):

    name = f'''portfolio_evolution_{config.model}_{config.currency}_{config.nb_classes}_
                {lower_bound}_{upper_bound}.png'''
    plt.figure(2)
    plt.plot(prediction['datetime'], prediction['budget'])
    plt.axhline(y=100, color='r', linestyle='-')
    plt.savefig(os.path.join(config.package_directory, 'models', name))
    plt.close()

# time_waw_list = [datetime.time(20,0,0), datetime.time(22,0,0)]
def check_hours(df, y_pred, date_start, date_end, lower_bound, upper_bound):
    
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)].copy()
    prediction.reset_index(inplace=True)
    prediction['y_pred'] = prediction['close'].shift(-config.steps_ahead) / prediction['close'] - 1
    prediction['change'] = [1 if y > 0 else 0 for y in prediction['y_pred']]
    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))

    prediction['datetime_london'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
        'Europe/London').dt.tz_localize(None)
    # make it so that time represents end of interval, not beginning - that's the time when transaction would be opened - prediction is for next hour
    prediction['datetime_london'] = prediction['datetime_london'] + pd.Timedelta(minutes=int(re.findall("\d+", config.freq)[0]))
    prediction['hour_london'] = prediction['datetime_london'].dt.time
    # prediction['trade'] = y_pred.argmax(axis=1)
    prediction['prediction'] = y_pred[:, 1]
    conditions = [
        (prediction['prediction'] <= lower_bound),
        (prediction['prediction'] > lower_bound) & (prediction['prediction'] <= upper_bound),
        (prediction['prediction'] > upper_bound)
    ]
    values = [0, 0.5, 1]
    prediction['trade'] = np.select(conditions, values)

    prediction['success'] = 0
    prediction.loc[((prediction['trade'] == 1) & (prediction['y_pred'] > 0)) |
                   ((prediction['trade'] == 0) & (prediction['y_pred'] < 0)), 'success'] = 1
    print('Distribution by hour')
    print(prediction.groupby('hour_london')['change'].mean())
    print('Distribution by hour for prediction')
    print(prediction.loc[prediction['trade'] != 0.5].groupby('hour_london').agg(count=('trade', 'size'),
                                                                             success=('success', 'mean')))
    return prediction