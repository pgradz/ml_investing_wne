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
        plt.text(j, i, round(matrix[i, j]/matrix[:,j].sum(),2),
                 horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(config.package_directory, 'models', 'confusion_matrix_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes)))


def compute_profitability_classes(df, y_pred, date_start, date_end):

    prediction = df.copy()
    prediction.reset_index(inplace=True)
    df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
    # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
    prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)]
    prediction['trade'] = y_pred.argmax(axis=1)
    prediction.reset_index(inplace=True)
    budget = 1000
    transaction = 'start'

    for i in range(prediction.shape[0]):
        if (prediction.loc[i, 'trade'] == config.nb_classes -1):
            # add transaction cost if position changes
            if transaction!='buy':
                budget = budget * (1-prediction.loc[i, 'cost'])
            transaction = 'buy'
            budget = budget + budget * prediction.loc[i, 'y_pred']
        elif (prediction.loc[i, 'trade'] == 0):
            # add transaction cost if position changes
            if transaction!='sell':
                budget = budget * (1-prediction.loc[i, 'cost'])
            transaction = 'sell'
            budget = budget + budget * (-prediction.loc[i, 'y_pred'])
        else:
            if transaction == 'start':
                continue
            elif transaction == 'buy':
                budget = budget + budget * prediction.loc[i, 'y_pred']
            else:
                budget = budget + budget * (- prediction.loc[i, 'y_pred'])
        prediction.loc[i, 'budget'] = budget
        prediction.loc[i, 'transaction'] = transaction

    plt.figure(2)
    plt.plot(prediction['datetime'], prediction['budget'])
    plt.axhline(y=1000, color='r', linestyle='-')
    plt.savefig(os.path.join(config.package_directory, 'models', 'portfolio_evolution_{}_{}_{}.png'.
                             format(config.model, config.currency, config.nb_classes)))

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
                no_of_rows_perc = round(prediction.loc[(prediction['y_prob'] > cutoff)].shape[0] / prediction.shape[0],3)
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
                no_of_rows_perc = round(prediction.loc[(prediction['y_prob'] < cutoff)].shape[0] / prediction.shape[0],3)
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

    return prediction.loc[i - 1, 'budget']
