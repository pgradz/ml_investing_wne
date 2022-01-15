import datetime

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import itertools
import ml_investing_wne.config as config

logger = logging.getLogger(__name__)


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

# #
# upper_bound = 0.5
# lower_bound = 0.5
# date_start = start_date
# date_end = end_date

def compute_profitability_classes(df, y_pred, date_start, date_end, lower_bound, upper_bound, time_waw_list=None):
    prediction = df.copy()
    prediction.reset_index(inplace=True)
    df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
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

    for i in range(prediction.shape[0]):
        if prediction.loc[i, 'trade'] == config.nb_classes - 1:
            # add transaction cost if position changes
            if transaction != 'buy':
                # if opposite transaction was open, first it has to be closed, so cost is incurred twice
                if not transaction:
                    budget = budget * (1 - prediction.loc[i, 'cost'])
                else:
                    budget = budget * (1 - (1 * prediction.loc[i, 'cost']))
            transaction = 'buy'
            budget = budget + budget * prediction.loc[i, 'y_pred']
        elif prediction.loc[i, 'trade'] == 0:
            # add transaction cost if position changes
            if transaction != 'sell':
                if not transaction:
                    budget = budget * (1 - prediction.loc[i, 'cost'])
                else:
                    budget = budget * (1 - (1 * prediction.loc[i, 'cost']))
            transaction = 'sell'
            budget = budget + budget * (-prediction.loc[i, 'y_pred'])
            # close transaction
        elif prediction.loc[i, 'trade'] == 0.5:
            if transaction in ['buy', 'sell']:
                #budget = budget * (1 - prediction.loc[i, 'cost'])
                transaction = None
        prediction.loc[i, 'budget'] = budget
        prediction.loc[i, 'transaction'] = transaction

    hits = prediction.loc[((prediction['transaction']=='buy') & (prediction['y_pred'] > 0)) |
                          ((prediction['transaction'] == 'sell') & (prediction['y_pred'] < 0))].shape[0]
    transactions = prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0]
    try:
        hits_ratio = hits/transactions
    except ZeroDivisionError:
        hits_ratio = 0
    share_of_time_active = round(prediction.loc[prediction['transaction'].isin(['buy', 'sell'])].shape[0]/prediction.shape[0],2)
    logger.info('share_of_time_active for bounds {}-{} is {} and hit ratio is {}'.format(lower_bound, upper_bound,
                                                                                         share_of_time_active, hits_ratio))
    plt.figure(2)
    plt.plot(prediction['datetime'], prediction['budget'])
    plt.axhline(y=100, color='r', linestyle='-')
    plt.savefig(os.path.join(config.package_directory, 'models', 'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes, lower_bound, upper_bound)))
    plt.close()

    logger.info('Portfolio result:  {}'.format(prediction.loc[i - 1, 'budget']))

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

    cutoff_df.to_csv(os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}.csv'.
                                  format(config.model, config.currency, config.nb_classes)), sep=";", decimal=",")

    return prediction.loc[i - 1, 'budget'], hits_ratio, share_of_time_active


# time_waw_list = [datetime.time(21,0,0), datetime.time(22,0,0), datetime.time(23,0,0)]
def check_hours(df, y_pred, date_start, date_end, lower_bound, upper_bound):
    prediction = df.copy()
    prediction.reset_index(inplace=True)
    df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
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


    prediction['success'] = 0
    prediction.loc[((prediction['trade'] == 1) & (prediction['y_pred'] > 0)) |
                          ((prediction['trade'] == 0) & (prediction['y_pred'] < 0)), 'success'] = 1
    prediction.loc[prediction['trade']==0].hour_waw.value_counts()
    prediction.loc[prediction['trade']==1].hour_waw.value_counts(normalize=True)
    print(prediction.loc[prediction['trade']!=0.5].groupby('hour_waw').agg(count=('trade', 'size'), success=('success', 'mean')))

# prediction.loc[prediction['hour_waw'].isin([datetime.time(20,0,0), datetime.time(22,0,0)])]