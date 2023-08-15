import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import logging
import os
import itertools
import re
import mlflow
import joblib
import tensorflow as tf
import importlib
from typing import Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import to_categorical
import ml_investing_wne.config as config
from ml_investing_wne.models import model_factory


pd.options.mode.chained_assignment = None


logger = logging.getLogger(__name__)
# from ml_investing_wne.utils import get_logger
# logger = get_logger()

class Experiment():
    def __init__(self, df, binarize_target=True, asset_factory=None, 
                 nb_classes=config.nb_classes, freq=config.freq, 
                 seq_len=config.seq_len, seq_stride=config.seq_stride,
                 train_end=config.train_end, val_end=config.val_end,
                 test_end=config.test_end, batch_size=config.batch,
                 seed=config.seed,budget=None) -> None:
        self.df = df
        self.binarize_target=binarize_target 
        self.asset_factory = asset_factory
        self.nb_classes=nb_classes
        self.freq=freq
        self.seq_len=seq_len
        self.seq_stride=seq_stride
        self.train_end=train_end
        self.val_end=val_end
        self.test_end=test_end
        self.batch_size=batch_size
        self.seed=seed
        # budget for performance evaluation
        if budget:
            self.budget = budget
        else:
            self.budget = 100

    def run(self):
        self.train_test_val_split()
        self.train_model()
        self.evaluate_model()


    def _train_test_val_split(self, df: pd.DataFrame,train_end, val_end,
                         test_end, seq_len) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Objective of this function is to split the data into train, test and validation sets in a manner that
        there is no overlap between the sets. It will remove some of the data at the end of train and val sets so
        test set is exactly what is expected and is comparable between different configurations'.cs/
        :param df: dataframe to split
        :return: train, test and validation sets, train date index, val date index, test date index
        '''
        train = df.loc[df.datetime < train_end] 
        # remove last seq_len rows from train
        train = train.iloc[:-seq_len]
        # update train_end
        train_end = train.iloc[-1]['datetime']
        # validation
        val = df.loc[(df.datetime > train_end) & (df.datetime < val_end)]
        val = val.iloc[:-seq_len]
        # update val_end
        val_end = val.iloc[-1]['datetime']
        # test
        test = df.loc[(df.datetime > val_end) & (df.datetime < test_end)]

        # xxx_date_index is needed to later get back datetime index
        train_date_index = train.reset_index()
        train_date_index = train_date_index[['index', 'datetime']]

        val_date_index = val.reset_index()
        val_date_index = val_date_index[['index', 'datetime']]

        test_date_index = test.reset_index()
        test_date_index = test_date_index[['index', 'datetime']]

        return train, val, test, train_date_index, val_date_index, test_date_index



    def train_test_val_split(self, sc_x=None):
        '''
        args:
            df: dataframe to split
            sc_x: scaler to use, if None, StandardScaler will be used. Option of passing scaler is needed for making ad-hoc predictions
        '''
        # columns to be dropped from training. y_pred is the target and datetime was carried for technical purposes. index and level_0 are just in case.
        COLUMNS_TO_DROP = ['y_pred', 'datetime', 'index', 'level_0', 'to_keep']

        df = self.df.copy()
        # take care if more than two classes
        if self.binarize_target:
            if self.nb_classes == 2:
                df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
            else:
                df['y_pred'] = pd.qcut(df['y_pred'], self.nb_classes, labels=range(self.nb_classes))

        # split train val test
        # move datetime from index to column
        df.reset_index(inplace=True)
        train, val, test, train_date_index, val_date_index, test_date_index = self._train_test_val_split(df, train_end=self.train_end, 
                                                                                                         val_end=self.val_end, test_end=self.test_end,
                                                                                                           seq_len=self.seq_len)
        train_y = train['y_pred']
        val_y = val['y_pred']
        test_y = test['y_pred']
        logger.info(f'balance of train set: {train_y.mean()}')
        logger.info(f'balance of val set: {val_y.mean()}')
        logger.info(f'balance of test set: {test_y.mean()}')
        if 'to_keep' in df.columns:
            train_keep = train['to_keep'].values.reshape(-1, 1)
            val_keep = val['to_keep'].values.reshape(-1, 1)
            test_keep = test['to_keep'].values.reshape(-1, 1)

        # drop columns
        for col in COLUMNS_TO_DROP:
            try:
                train.drop(columns=[col], inplace=True)
                val.drop(columns=[col], inplace=True)
                test.drop(columns=[col], inplace=True)
            except:
                pass
       
        # needed to pass later to the model as dim of the input
        self.no_features = train.shape[1]

        # scaler, if not passed in the function, has to be fit on train set, it's easier to do it here
        if not sc_x:
            sc_x = StandardScaler()
            train = sc_x.fit_transform(train)
        else:
            train = sc_x.transform(train)
        self.sc_x = sc_x
        val = sc_x.transform(val)
        test = sc_x.transform(test)
        joblib.dump(sc_x, os.path.join(config.package_directory, 'models',
                                    'sc_x_{}_{}.save'.format(config.currency, self.freq)))
        if 'to_keep' in df.columns:
            train = np.hstack((train_keep, train))
            val = np.hstack((val_keep, val))
            test = np.hstack((test_keep, test))
        
        # create tensorflow datasets
        # train
        train_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=to_categorical(train_y.values[self.seq_len-1:]),
                                                                sequence_length=self.seq_len, sequence_stride=self.seq_stride, batch_size=None)
        if 'to_keep' in df.columns:
            train_dataset = self.filter_tf_dataset(train_dataset, self.batch_size, self.seq_len, filter_rows=True)
        self.train_dataset = train_dataset.batch(self.batch_size)
        
        train_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=train_date_index.values[self.seq_len-1:, 0].astype(int),
                                                                sequence_length=self.seq_len, sequence_stride=self.seq_stride, batch_size=None)
        if 'to_keep' in df.columns:
            train_date_index_dataset = self.filter_tf_dataset(train_date_index_dataset, self.batch_size, self.seq_len, filter_rows=True)
        train_date_index_dataset = train_date_index_dataset.batch(self.batch_size)

        # val
        val_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=to_categorical(val_y.values[self.seq_len-1:]), 
                                                                sequence_length=self.seq_len, sequence_stride=1, batch_size=None)
        if 'to_keep' in df.columns:
            val_dataset = self.filter_tf_dataset(val_dataset, self.batch_size, self.seq_len, filter_rows=False)
        self.val_dataset = val_dataset.batch(self.batch_size)
    
        val_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=val_date_index.values[self.seq_len-1:, 0].astype(int),
                                                                sequence_length=self.seq_len, sequence_stride=1, batch_size=None)
        if 'to_keep' in df.columns:
            val_date_index_dataset = self.filter_tf_dataset(val_date_index_dataset, self.batch_size, self.seq_len, filter_rows=False)
        val_date_index_dataset = val_date_index_dataset.batch(self.batch_size)

        # test
        test_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=to_categorical(test_y.values[self.seq_len-1:]),
                                                                sequence_length=self.seq_len, sequence_stride=1, batch_size=None)
        if 'to_keep' in df.columns:
            test_dataset = self.filter_tf_dataset(test_dataset, self.batch_size, self.seq_len, filter_rows=False)
        self.test_dataset = test_dataset.batch(self.batch_size)

        test_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=test_date_index.values[self.seq_len-1:, 0].astype(int),
                                                                sequence_length=self.seq_len, sequence_stride=1, batch_size=None)
        if 'to_keep' in df.columns:
            test_date_index_dataset = self.filter_tf_dataset(test_date_index_dataset, self.batch_size, self.seq_len, filter_rows=False)
        test_date_index_dataset = test_date_index_dataset.batch(self.batch_size)


        # # create tensorflow datasets
        # self.train_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=to_categorical(train_y.values[seq_len-1:]),
        #                                                         sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)                                                            
        # train_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=train_date_index.values[seq_len-1:, 0].astype(int),
        #                                                         sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
        # self.val_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=to_categorical(val_y.values[seq_len-1:]), 
        #                                                         sequence_length=seq_len, sequence_stride=1, batch_size=batch_size)
        # val_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=val_date_index.values[seq_len-1:, 0].astype(int),
        #                                                         sequence_length=seq_len, sequence_stride=1, batch_size=batch_size)
        # self.test_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=to_categorical(test_y.values[seq_len-1:]),
        #                                                         sequence_length=seq_len, sequence_stride=1, batch_size=batch_size)
        # test_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=test_date_index.values[seq_len-1:, 0].astype(int),
        #                                                         sequence_length=seq_len, sequence_stride=1, batch_size=batch_size)

        # get back date indices
        self.train_date_index = self.get_datetime_indices(train_date_index_dataset, train_date_index)
        self.val_date_index = self.get_datetime_indices(val_date_index_dataset, val_date_index)
        self.test_date_index = self.get_datetime_indices(test_date_index_dataset, test_date_index)

        # shuffle training data set
        self.train_dataset.train_dataset = self.train_dataset.shuffle(buffer_size=train.shape[0], reshuffle_each_iteration=True)

        return None
    
    def filter_tf_dataset(self, tf_dataset, batch_size, seq_len, filter_rows=True):
        ''' This function is used to filter the tensorflow dataset
        args:
            tf_dataset: tensorflow dataset, unbatched. First column contains info for filtering
            batch_size: batch size
            seq_len: sequence length
            to_keep_index: index of the column to keep
            filter_rows: whether to filter rows or not
        return:
            tf_dataset: filtered tensorflow dataset
        '''
        tf_dataset_size = self.tf_dataset_size(tf_dataset)
        logger.info('tf dataset size: {}'.format(tf_dataset_size))
        # if the last obs in the array (the latest row) is not 1, then remove the sequence
        if filter_rows:
            tf_dataset = tf_dataset.filter(lambda x, _: x[seq_len-1,0] == 1)
        # remove the first column
        tf_dataset = tf_dataset.map(lambda x,y : (x[:, 1: ], y))
        tf_dataset_size = self.tf_dataset_size(tf_dataset)
        logger.info('tf dataset size after filtering: {}'.format(tf_dataset_size))
        return tf_dataset
    
    @staticmethod
    def tf_dataset_size(tf_dataset: tf.data.Dataset):
        ''' This function is used to get the size of the tensorflow dataset
        args:
            tf_dataset: tensorflow dataset
        return:
            size: size of the dataset
        '''
        return len(list(tf_dataset.as_numpy_iterator()))

    def set_y_true(self, dataset: tf.data.Dataset):
        ''' This function is used to get true labels from the tensorflow dataset
        args:
            dataset: tensorflow dataset
        return:
            y_true: true labels
        '''

        y_true = np.empty((0, 2)) # store true labels
        # iterate over the dataset
        for x, label_batch in dataset:  
            # append true labels
            y_true = np.concatenate((y_true, label_batch.numpy()), axis=0)

        return y_true[:, 1]

    
    def get_datetime_indices(self, date_index_dataset: tf.data.Dataset, date_index: pd.DataFrame) -> pd.DataFrame:
        '''
        This function is used to get back datetime index from the dataset that has the same shape as original dataframes and 
        can be concatenated with them

        args:
            date_index_dataset: dataset with date indices
            date_index: original date index
        return:
            date_index: date index with the same shape as original (train, val, test) dataframe
        '''
        for i, batch in enumerate(date_index_dataset):
            if i == 0:
                indices = batch[1].numpy()
            else:
                indices =  np.concatenate((indices, batch[1].numpy()), axis=0)

        date_index = date_index.loc[date_index['index'].isin(indices)]
        date_index.reset_index(inplace=True, drop=True)

        return date_index
    
    def hyperparameter_tunning(self):
        # load model tunner dynamically
        MyHyperModel = getattr(importlib.import_module(f'ml_investing_wne.tf_models.{config.model}'),'MyHyperModel')
        my_hyper_model = MyHyperModel(input_shape=(self.seq_len, self.no_features), train_dataset=self.train_dataset, val_dataset=self.val_dataset,
                                      seed=self.seed)
        my_hyper_model.run_tuner()
        self.model = my_hyper_model.get_best_model()

    def set_budget(self, budget):
        self.budget = budget

    def get_budget(self):
        return self.budget
    
    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
    
    def get_hit_counter(self):
        return self.hit_counter
    
    def get_trades_counter(self):
        return self.trades_counter
    
    def get_ml_flow_experiment_name(self):
        """creates mlflow experiment name based on config parameters.

        Returns:
            str: mlflow experiment name
        """

        ml_flow_experiment_name = (f"{config.provider}_{config.model}_{str(config.nb_classes)}"
                                    f"{config.freq}_{str(config.steps_ahead)}_{str(config.seq_len)}")
        return ml_flow_experiment_name


    def get_training_model_path(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        model_name = f"{config.model}_{config.provider}_{config.currency}_{config.freq}_{config.steps_ahead}.h5"
        model_path_training = os.path.join(config.package_directory, 'models', model_name)

        return model_path_training

    def get_xtb_training_model_path(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        model_name = f"{config.model}_hist_data_{config.currency}_{config.freq}_{config.steps_ahead}.h5"
        model_path_training = os.path.join(config.package_directory, 'models', model_name)

        return model_path_training

    def get_final_model_path(self):

        model_name = (f"{config.model}_{config.provider}_{config.currency}_{config.freq}_"
                        f"{str(config.steps_ahead)}_{config.seq_len}")
        model_path_final = os.path.join(config.package_directory, 'models', 'production', model_name)
        return model_path_final

    def get_callbacks(self):
        
        if config.provider == 'xtb':
            training_model_path = self.get_xtb_training_model_path()
        else:
            training_model_path = self.get_training_model_path()

        early_stop = EarlyStopping(monitor='val_accuracy', patience=config.patience, 
                    restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=training_model_path, monitor='val_accuracy',
                            verbose=1, save_best_only=True)
        csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), 
                                append=True, separator=';')
        callbacks = [early_stop, model_checkpoint, csv_logger]

        return callbacks

    def get_scaler(self):
        
        sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                    f'sc_x_{config.currency}_{config.freq}.save'))
        return sc_x


    def train_model(self):
        mlflow.tensorflow.autolog()
        mlflow.set_experiment(experiment_name=self.get_ml_flow_experiment_name())
        callbacks = self.get_callbacks()
        if self.model is None:
            self.model = model_factory(input_shape=(self.seq_len, self.no_features))
        self.history = self.model.fit(self.train_dataset, batch_size=config.batch, epochs=config.epochs, verbose=2,
                            validation_data=self.val_dataset, callbacks=callbacks)
        self.model.save(self.get_final_model_path())

    def evaluate_model(self):

        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        logger.info('Test accuracy : %.4f', test_acc)
        logger.info('Test loss : %.4f', test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.set_tag('currency', config.currency)
        mlflow.set_tag('frequency', config.freq)
        mlflow.set_tag('steps_ahead', config.steps_ahead)
        # mlflow.log_metric('y_distribution', self.y.mean())
        # mlflow.log_metric('y_val_distribution', self.y_val.mean())
        # mlflow.log_metric('y_test_distribution', self.y_test.mean())
        mlflow.log_metric('cost', config.cost)
        mlflow.log_metric('seq_len', config.seq_len)

        self.y_val = self.set_y_true(self.val_dataset)
        y_pred_val = self.model.predict(self.val_dataset)
        y_pred_class_val = [1 if y > 0.5 else 0 for y in y_pred_val[:,1]]

        self.y_test = self.set_y_true(self.test_dataset)
        y_pred_test = self.model.predict(self.test_dataset)
        y_pred_class_test = [1 if y > 0.5 else 0 for y in y_pred_test[:,1]]

        # roc_auc = roc_auc_score(self.y_test, y_pred_class)
        # f1 = f1_score(self.y_test, y_pred_class)
        # logger.info('roc_auc : %.4f',roc_auc)
        # logger.info('f1 : %.4f', f1)
        # mlflow.log_metric('roc_auc', roc_auc)
        # mlflow.log_metric('f1', f1)

        accuracy = (self.y_test == y_pred_class_test).mean()
        logger.info('accuracy on test set : %.4f', accuracy)
        mlflow.log_metric('accuracy', accuracy)
        
        accuracy_val = (self.y_val == y_pred_class_val).mean()
        logger.info('accuracy on validation set : %.4f', accuracy_val)
        mlflow.log_metric('accuracy_val', accuracy_val)

        # validation and test data will be matched based on index
        self.df = self.df.assign(index=range(len(self.df)))
        self.train_date_index['train_val_test'] = 'train'
        # prediction for train set won't be needed
        self.train_date_index['prediction'] = -1
        self.val_date_index['train_val_test'] = 'val'
        self.val_date_index['prediction'] = y_pred_val[:, 1]
        self.test_date_index['train_val_test'] = 'test'
        self.test_date_index['prediction'] = y_pred_test[:,1]
        train_val_test = pd.concat([self.train_date_index, self.val_date_index, self.test_date_index], axis=0)
        self.df = self.df.merge(train_val_test, on='index', how='left')
        self.df = self.add_cost(self.df)

        lower_bounds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        upper_bounds = [1 - lower for lower in lower_bounds]
        logger.info('Prediction accuracy for validation set')
        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            self.accuracy_by_threshold(y_pred_val, self.y_val, lower_bound, upper_bound)
        logger.info('Prediction accuracy for test set')
        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            self.accuracy_by_threshold(y_pred_test, self.y_test, lower_bound, upper_bound)
        df_eval = self.df.merge(self.df_3_barriers_additional_info, on='datetime', how='outer')

        df_eval_test = df_eval.loc[df_eval.train_val_test == 'test']

        self.budget, self.hit_counter, self.trades_counter = self.backtest(df_eval_test, 0.4, 0.6)


    def backtest(self, df, lower_bound, upper_bound):
        ''' This function is used to backtest the model
        args:
            df: dataframe with predictions and actual values
            lower_bound: lower bound for the prediction
            upper_bound: upper bound for the prediction
        return:
        '''
        df.reset_index(inplace=True, drop=True)
        conditions = [
            (df['prediction'] <= lower_bound),
            (df['prediction'] > lower_bound) & (df['prediction'] <= upper_bound),
            (df['prediction'] > upper_bound)
        ]
        values = [0, 0.5, 1]

        df['trade'] = np.select(conditions, values)
        # INITIALIZE PORTFOLIO
        budget = self.budget
        logger.info('Starting trading with:  %.2f', budget)
        transaction = 'No trade'
        i = 0
        df = df[['datetime','open', 'close', 'high','low', 'prc_change', 'cost', 'trade', 'time_step', 'prediction','barrier_touched', 'barrier_touched_date', 'bottom_barrier', 'top_barrier']]
        while i < df.shape[0]:
        
            if df.loc[i, 'trade'] == 1:
                if transaction != 'No trade':
                    # first close open position if open
                    budget = budget * (1 - df.loc[i, 'cost']) 
                # then open new position
                budget = budget * (1 - df.loc[i, 'cost'])
                transaction = 'buy'
                budget = budget + budget * df.loc[i, 'prc_change']
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + df.loc[i, 'time_step'] # jump till barrier is touched
            elif df.loc[i, 'trade'] == 0:
                # add transaction cost if position changes
                if transaction != 'No trade':
                    # first close open position if open
                    budget = budget * (1 - df.loc[i, 'cost']) 
                 # then open new position
                budget = budget * (1 - df.loc[i, 'cost'])
                transaction = 'sell'
                budget = budget + budget * (-df.loc[i, 'prc_change'])
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + df.loc[i, 'time_step'] # jump till barrier is touched
            elif df.loc[i, 'trade'] == 0.5:
                if transaction in ['buy', 'sell']:
                    budget = budget * (1 - df.loc[i, 'cost']) # add cost while closing position
                    transaction = 'No trade'
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + 1

        # temporary for building sharpe ratio
        df.to_csv(os.path.join(config.package_directory, 'models',
                                f'''backtest_{config.currency}_{config.RUN_SUBTYPE}_{config.val_end.strftime("%Y%m%d")}_{config.test_end.strftime("%Y%m%d")}_{config.seed}.csv'''))
        # SUMMARIZE RESULTS
        hits = df.loc[((df['transaction'] == 'buy') & (df['prc_change'] > 0)) |
                            ((df['transaction'] == 'sell') & (df['prc_change'] < 0))].shape[0]
        transactions = df.loc[df['transaction'].isin(['buy', 'sell'])].shape[0]
        try:
            hits_ratio = hits / transactions
        except ZeroDivisionError:
            hits_ratio = 0

        logger.info('Portfolio result:  %.2f', budget)
        logger.info('Hit ratio:  %.2f', hits_ratio)

        self.plot_portfolio(df, lower_bound, upper_bound)

        return budget, hits, transactions

    def accuracy_by_threshold(self, y_pred, actual, lower_bound, upper_bound):
        '''calculate accuracy for a given upper and lowe threshold
        y_pred: array of predictions [2d array]
        actual: array of actual values [1d array]
        lower_bound: lower bound of threshold
        upper_bound: upper bound of threshold'''

        y_pred_class = [1 if y > 0.5 else 0 for y in y_pred[:,1]]
        df = pd.DataFrame({'prediction': y_pred[:,1], 'y_true': actual, 'y_pred': y_pred_class})
        predictions_above_threshold = df.loc[(df['prediction'] < lower_bound) | (df['prediction'] > upper_bound)]
        accuracy = (predictions_above_threshold['y_true'] == predictions_above_threshold['y_pred']).mean()
        logger.info('accuracy for threshold between %.2f and %.2f : %.4f, based on %.1f observations', lower_bound, upper_bound, accuracy, predictions_above_threshold.shape[0]/df.shape[0])

        return None

    def load_test_dates(self):

        name = f'test_{config.currency}_{config.freq}.save'

        start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                            f'first_sequence_ends_{name}'))
        end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                            f'last_sequence_ends_{name}'))

        return start_date, end_date

    def add_cost(self, df):
        if config.RUN_TYPE == 'forex':
            if 'JPY' in config.currency:
                df['cost'] = (config.cost / 100) / df['close']
            else:
                df['cost'] = (config.cost / 10000) / df['close']
        elif config.RUN_TYPE == 'crypto':
            df['cost']  = config.cost
        else:
            logger.info('Did not find cost information, assuming 0 costs')
            df['cost'] = 0
        return df


    def compute_profitability_classes(self, df, y_pred, date_start, date_end, lower_bound, upper_bound,
                                    hours_exclude=None):

       
        if config.RUN_SUBTYPE == 'triple_barrier_time_aggregated':
            df = df.merge(self.asset_factory.df_3_barriers_additional_info[['datetime', 'prc_change']], on='datetime', how='inner')
            # TODO: check that y_pred here is on the same scale as in another flow
            df['y_pred'] = df['prc_change']
        else:
        # recreate target as continous variable
            df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close'] - 1
        # new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
        prediction = df.loc[(df.datetime >= date_start) & (df.datetime <= date_end)].copy()
        if config.provider == 'hist_data':
            prediction['datetime_local'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
                'Europe/London').dt.tz_localize(None)
        else:
            prediction['datetime_local'] = prediction['datetime']

        prediction['hour_local'] = prediction['datetime_local'].dt.time
        # TODO: here triple barriers fails
        prediction = prediction.loc[prediction.datetime.isin(self.test_y_index)]
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
            if budget is None:
                print('is none')

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

        self.plot_portfolio(prediction, lower_bound, upper_bound)
        
        return budget, hits_ratio, share_of_time_active

    def plot_portfolio(self, prediction, lower_bound, upper_bound):

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
