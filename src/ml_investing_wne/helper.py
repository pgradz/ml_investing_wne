from collections import namedtuple
import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import itertools
import re
import mlflow
import joblib
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import ml_investing_wne.config as config

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def get_ml_flow_experiment_name():

    ml_flow_experiment_name = f'''hist_data_{config.model}_{str(config.nb_classes)}_
                                  {config.freq}_{str(config.steps_ahead)}_{str(config.seq_len)}'''
    return ml_flow_experiment_name


def get_training_model_path():

    model_name = f'''{config.model}_hist_data_{config.currency}_{config.freq}_
                     {config.steps_ahead}.h5'''
    model_path_training = os.path.join(config.package_directory, 'models', model_name)

    return model_path_training

def get_final_model_path():

    model_name = '''{config.model}_hist_data_{config.currency}_{config.freq}_
                    {str(config.steps_ahead)}_{config.seq_len}'''
    model_path_final = os.path.join(config.package_directory, 'models', 'production', model_name)
    return model_path_final

def get_callbacks():

    early_stop = EarlyStopping(monitor='val_accuracy', patience=config.patience, 
                restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=get_training_model_path(), monitor='val_accuracy',
                        verbose=1, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), 
                            append=True, separator=';')
    callbacks = [early_stop, model_checkpoint, csv_logger]

    return callbacks

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
    
    # PREPARE DATASET
    prediction = df.copy()
    prediction.reset_index(inplace=True)
    # recreate target as continous variable
    df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close'] - 1
    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)]
    prediction['datetime_local'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
        'Europe/London').dt.tz_localize(None)
    prediction['hour_local'] = prediction['datetime_local'].dt.time
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
    logger.info('Portfolio result:  %d', budget)

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


# prediction.loc[prediction['hour_waw'].isin([datetime.time(20,0,0), datetime.time(22,0,0)])]

def check_volatility(df, y_pred, date_start, date_end, lower_bound, upper_bound, sell_limit, time_waw_list=None):
    
    df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close'] - 1
    df['close_close'] = df['close'].shift(-1) - df['close']
    df['high_close'] = df['high'].shift(-1) - df['close']
    df['low_close'] = df['low'].shift(-1) - df['close']
    df['close_close'] = df['close_close'].round(4)
    df['high_close'] = df['high_close'].round(4)
    df['low_close'] = df['low_close'].round(4)

    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)]
    prediction['datetime_waw'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
        'Europe/Warsaw').dt.tz_localize(None)
    prediction['hour_waw'] = prediction['datetime_waw'].dt.time
    # prediction['trade'] = y_pred.argmax(axis=1)
    prediction['prediction'] = y_pred[:, 1]
    conditions = [
        (prediction['prediction'] <= lower_bound),
        (prediction['prediction'] > lower_bound) & (prediction['prediction'] <= upper_bound),
        (prediction['prediction'] > upper_bound)
    ]
    values = [0, 0.5, 1]
    prediction['trade'] = np.select(conditions, values)
    if time_waw_list:
        prediction.loc[~prediction['hour_waw'].isin(time_waw_list), 'trade'] = 0.5

    buy = prediction.loc[prediction['trade'] == 1]
    print('buy high close \n', buy['high_close'].describe())
    print('buy close close \n', buy['close_close'].describe())

    sell = prediction.loc[prediction['trade'] == 0]
    print('sell low close \n', sell['low_close'].describe())
    print('sell close close \n', sell['close_close'].describe())

    prediction.reset_index(inplace=True)

    budget = 100
    transaction = None
    sell_limit = sell_limit / 10000
    for i in range(prediction.shape[0]):
        if prediction.loc[i, 'trade'] == config.nb_classes - 1:
            # add transaction cost if position changes
            if transaction != 'buy':
                budget = budget * (1 - prediction.loc[i, 'cost'])
                transaction = 'buy'
            if prediction.loc[i, 'high_close'] > sell_limit:
                budget = budget + budget * sell_limit / prediction.loc[i, 'close']
            else:
                budget = budget + budget * prediction.loc[i, 'y_pred']
        elif prediction.loc[i, 'trade'] == 0:
            # add transaction cost if position changes
            if transaction != 'sell':
                budget = budget * (1 - prediction.loc[i, 'cost'])
                transaction = 'sell'
            if prediction.loc[i, 'low_close'] < -sell_limit:
                budget = budget + budget * sell_limit / prediction.loc[i, 'close']
            else:
                budget = budget + budget * (-prediction.loc[i, 'y_pred'])
            # close transaction
        elif prediction.loc[i, 'trade'] == 0.5:
            if transaction in ['buy', 'sell']:
                # budget = budget * (1 - prediction.loc[i, 'cost'])
                transaction = None
        prediction.loc[i, 'budget'] = budget
        prediction.loc[i, 'transaction'] = transaction

    hits = prediction.loc[((prediction['transaction'] == 'buy') & (prediction['y_pred'] > 0)) |
                          ((prediction['transaction'] == 'sell') & (prediction['y_pred'] < 0))].shape[0]
    transactions = prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0]
    try:
        hits_ratio = hits / transactions
    except ZeroDivisionError:
        hits_ratio = 0
    share_of_time_active = round(
        prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0] / prediction.shape[0], 2)
    logger.info('share_of_time_active for bounds {}-{} is {} and hit ratio is {}'.format(lower_bound, upper_bound,
                                                                                         share_of_time_active,
                                                                                         hits_ratio))
    plt.figure(2)
    plt.plot(prediction['datetime'], prediction['budget'])
    plt.axhline(y=100, color='r', linestyle='-')
    plt.savefig(os.path.join(config.package_directory, 'models', 'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes, lower_bound, upper_bound)))
    plt.close()

    logger.info('Portfolio result:  {}'.format(prediction.loc[i - 1, 'budget']))

    return prediction.loc[i - 1, 'budget'], hits_ratio, share_of_time_active



def compute_profitability_classes_fixed_percentage(df, y_pred, date_start, date_end, lower_bound, upper_bound, time_waw_list=None):
    prediction = df.copy()
    prediction.reset_index(inplace=True)
    df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close'] - 1
    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)]
    prediction['datetime_waw'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
        'Europe/Warsaw').dt.tz_localize(None)
    prediction['hour_waw'] = prediction['datetime_waw'].dt.time
    # prediction['trade'] = y_pred.argmax(axis=1)
    prediction['prediction'] = y_pred[:, 1]
    conditions = [
        (prediction['prediction'] <= lower_bound),
        (prediction['prediction'] > lower_bound) & (prediction['prediction'] <= upper_bound),
        (prediction['prediction'] > upper_bound)
    ]
    values = [0, 0.5, 1]
    prediction['trade'] = np.select(conditions, values)
    if time_waw_list:
        prediction.loc[~prediction['hour_waw'].isin(time_waw_list), 'trade'] = 0.5
    prediction.reset_index(inplace=True)
    budget = 100
    transaction = None
    i = 0
    while i < prediction.shape[0]:
        # for i in range(prediction.shape[0]):
        if prediction.loc[i, 'trade'] == config.nb_classes - 1:
            # add transaction cost if position changes
            if transaction != 'buy':
                # initally I assumed that cost can be beared twice, but with spread as only cost it should count only once
                if not transaction:
                    budget = budget * (1 - prediction.loc[i, 'cost'])
                else:
                    budget = budget * (1 - prediction.loc[i, 'cost'])
            transaction = 'buy'
            budget = budget + budget * prediction.loc[i, 'y_pred']
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + config.steps_ahead
        elif prediction.loc[i, 'trade'] == 0:
            # add transaction cost if position changes
            if transaction != 'sell':
                if not transaction:
                    budget = budget * (1 - prediction.loc[i, 'cost'])
                else:
                    budget = budget * (1 - (1 * prediction.loc[i, 'cost']))
            transaction = 'sell'
            budget = budget + budget * (-prediction.loc[i, 'y_pred'])
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + config.steps_ahead
        elif prediction.loc[i, 'trade'] == 0.5:
            if transaction in ['buy', 'sell']:
                # budget = budget * (1 - prediction.loc[i, 'cost'])
                transaction = None
            prediction.loc[i, 'budget'] = budget
            prediction.loc[i, 'transaction'] = transaction
            i = i + 1

    hits = prediction.loc[((prediction['transaction'] == 'buy') & (prediction['y_pred'] > 0)) |
                          ((prediction['transaction'] == 'sell') & (prediction['y_pred'] < 0))].shape[0]
    transactions = prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0]
    try:
        hits_ratio = hits / transactions
    except ZeroDivisionError:
        hits_ratio = 0
    share_of_time_active = round(prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0] * \
                                 config.steps_ahead / prediction.shape[0], 2)
    logger.info('share_of_time_active for bounds {}-{} is {} and hit ratio is {}'.format(lower_bound, upper_bound,
                                                                                         share_of_time_active,
                                                                                         hits_ratio))
    plt.figure(2)
    plt.plot(prediction['datetime'], prediction['budget'])
    plt.axhline(y=100, color='r', linestyle='-')
    plt.savefig(os.path.join(config.package_directory, 'models', 'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes, lower_bound, upper_bound)))
    plt.close()

    logger.info('Portfolio result:  {}'.format(budget))

    # cut off analysis starts here
    cutoffs = [0.9, 0.8, 0.75, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.25, 0.2, 0.1]
    prediction['y_prob'] = y_pred[:, 1]
    cutoff_df = pd.DataFrame(
        columns=['currency', 'probability_cutoff', 'no_of_rows', 'no_of_rows_perc', 'correct_predictions',
                 'correct_predictions_perc', 'prod_result'])

    for cutoff in cutoffs:
        if cutoff > 0.5:
            no_of_rows = prediction.loc[prediction['y_prob'] > cutoff].shape[0]
            try:
                no_of_rows_perc = round(prediction.loc[(prediction['y_prob'] > cutoff)].shape[0] / prediction.shape[0],
                                        3)
            except ZeroDivisionError:
                no_of_rows_perc = 0
            correct_predictions = prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > cutoff)].shape[0]
            try:
                correct_predictions_perc = round(correct_predictions / no_of_rows, 3)
            except ZeroDivisionError:
                correct_predictions_perc = 0
            prod = prediction.loc[(prediction['y_prob'] > cutoff)]['y_pred'] + 1
            prod_result = round(np.prod(list(prod)), 3)
        else:
            no_of_rows = prediction.loc[prediction['y_prob'] < cutoff].shape[0]
            try:
                no_of_rows_perc = round(prediction.loc[(prediction['y_prob'] < cutoff)].shape[0] / prediction.shape[0],
                                        3)
            except ZeroDivisionError:
                no_of_rows_perc = 0
            correct_predictions = prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < cutoff)].shape[0]
            try:
                correct_predictions_perc = round(correct_predictions / no_of_rows, 3)
            except ZeroDivisionError:
                correct_predictions_perc = 0
            prod = prediction.loc[(prediction['y_prob'] < cutoff)]['y_pred'] + 1
            prod_result = round(np.prod(list(prod)), 3)

        row = {'currency': config.currency,
               'probability_cutoff': cutoff,
               'no_of_rows': no_of_rows,
               'no_of_rows_perc': no_of_rows_perc,
               'correct_predictions': correct_predictions,
               'correct_predictions_perc': correct_predictions_perc,
               'prod_result': prod_result
               }
        cutoff_df = cutoff_df.append(row, ignore_index=True)

    cutoff_df.to_csv(os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}_{}.csv'.
                                  format(config.model, config.currency, config.nb_classes, config.steps_ahead)), sep=";", decimal=",")

    return budget, hits_ratio, share_of_time_active
