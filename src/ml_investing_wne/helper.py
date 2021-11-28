from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
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
            transaction = 'buy'
            budget = budget + budget * prediction.loc[i, 'y_pred']
        elif (prediction.loc[i, 'trade'] == 0):
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

    return prediction.loc[i - 1, 'budget']
