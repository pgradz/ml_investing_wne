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
    def __init__(self, df, binarize_target=True, time_step=None, asset_factory=None) -> None:
        self.df = df
        self.binarize_target=binarize_target 
        self.time_step=time_step
        self.asset_factory = asset_factory

    def run(self):
        self.train_test_val_split()
        self.train_model()
        self.evaluate_model()


    def _train_test_val_split(self, df: pd.DataFrame,train_end, val_end,
                         test_end, seq_len) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
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



    def train_test_val_split(self, sc_x=None, nb_classes=config.nb_classes, freq=config.freq,
                            seq_len=config.seq_len, seq_stride=config.seq_stride,
                            train_end=config.train_end, val_end=config.val_end,
                            test_end=config.test_end, batch_size=config.batch):
        '''
        args:
            df: dataframe to split
            sc_x: scaler to use, if None, StandardScaler will be used. Option of passing scaler is needed for making ad-hoc predictions
            nb_classes: number of classes to predict
            freq: frequency of the data
            seq_len: length of the sequence
            seq_stride: stride of the sequence
            steps_ahead: how many steps ahead to predict
            train_end: end of the train set
            val_end: end of the validation set  
            test_end: end of the test set
        '''
        # columns to be dropped from training. y_pred is the target and datetime was carried for technical purposes. index and level_0 are just in case.
        COLUMNS_TO_DROP = ['y_pred', 'datetime', 'index', 'level_0']

        df = self.df.copy()
        # take care if more than two classes
        if self.binarize_target:
            if nb_classes == 2:
                df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
            else:
                df['y_pred'] = pd.qcut(df['y_pred'], nb_classes, labels=range(nb_classes))

        # split train val test
        # move datetime from index to column
        df.reset_index(inplace=True)
        train, val, test, train_date_index, val_date_index, test_date_index = self._train_test_val_split(df, train_end=train_end, val_end=val_end, test_end=test_end, seq_len=seq_len)
        train_y = train['y_pred']
        val_y = val['y_pred']
        test_y = test['y_pred']
        # drop columns
        for col in COLUMNS_TO_DROP:
            try:
                train.drop(columns=[col], inplace=True)
                val.drop(columns=[col], inplace=True)
                test.drop(columns=[col], inplace=True)
            except:
                pass
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
                                    'sc_x_{}_{}.save'.format(config.currency, freq)))

        # store data shape
        self.no_features = train.shape[1]
        self.seq_len = seq_len

        # create tensorflow datasets
        self.train_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=train_y.values[seq_len-1:],
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)                                                            
        train_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train, targets=train_date_index.values[seq_len-1:, 0].astype(int),
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
        self.val_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=val_y.values[seq_len-1:], 
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
        val_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val, targets=val_date_index.values[seq_len-1:, 0].astype(int),
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
        self.test_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=test_y.values[seq_len-1:],
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
        test_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test, targets=test_date_index.values[seq_len-1:, 0].astype(int),
                                                                sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)

        # get back date indices
        self.train_date_index = self.get_datetime_indices(train_date_index_dataset, train_date_index)
        self.val_date_index = self.get_datetime_indices(val_date_index_dataset, val_date_index)
        self.test_date_index = self.get_datetime_indices(test_date_index_dataset, test_date_index)

        return None

    def set_y_true(self, dataset: tf.data.Dataset):
        ''' This function is used to get true labels from the tensorflow dataset
        args:
            dataset: tensorflow dataset
        return:
            y_true: true labels
        '''

        y_true = np.array([])  # store true labels
        # iterate over the dataset
        for x, label_batch in dataset:  
            # append true labels
            y_true = np.concatenate((y_true, label_batch.numpy()), axis=0)

        return y_true

    
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


    # prepare sequences
    def split_sequences(self, sequences_x, sequences_y, n_steps, datetime_series, steps_ahead, name, 
                        time_step=None):
        X, y, y_index = list(), list(), list()
        i = 0
        jump = 0
        while  i  <  len(sequences_x)+1:
            
            # find the end of this pattern
            end_ix = i + n_steps
            # print('i ', i, datetime_series[i])
            # print('end_ix  ', end_ix,  datetime_series[end_ix])
            if i == 0:
                logger.info('first sequence begins: {}'.format(datetime_series[i]))
                logger.info('first sequence ends: {}'.format(datetime_series[end_ix-1]))
                joblib.dump(datetime_series[end_ix-1], os.path.join(config.package_directory, 'models',
                                            'first_sequence_ends_{}_{}_{}.save'.format(name, config.currency, config.freq)))
            # check if we are beyond the dataset
            if end_ix + steps_ahead > len(sequences_x) + 1:
                logger.info('last sequence begins: {}'.format(datetime_series[i-jump]))
                logger.info('last sequence ends: {}'.format(datetime_series[end_ix-jump-1]))
                joblib.dump(datetime_series[end_ix-jump-1], os.path.join(config.package_directory, 'models',
                                            'last_sequence_ends_{}_{}_{}.save'.format(name, config.currency, config.freq)))
                break
            # gather input and output parts of the pattern
            seq_x, seq_y, seq_y_index = sequences_x[i:end_ix,:], sequences_y[end_ix - 1], datetime_series[end_ix-1]
            X.append(seq_x)
            y.append(seq_y)
            y_index.append(seq_y_index)
            if isinstance(time_step, pd.Series):
                jump = time_step[end_ix - 1]
            else:
                jump = 1
            i+=jump
        if name == 'train':
            self.train_y_index = y_index
        elif name == 'val':
            self.val_y_index = y_index
        elif name == 'test':
            self.test_y_index = y_index

        return np.array(X), np.array(y)


    def train_test_val_split_deprecated(self, sc_x=None, nb_classes=config.nb_classes, freq=config.freq,
                            seq_len=config.seq_len, steps_ahead=config.steps_ahead,
                            train_end=config.train_end, val_end=config.val_end,
                            test_end=config.test_end):

        df = self.df.copy()
        colunns_to_drop = ['y_pred', 'datetime', 'index', 'level_0']
        minutes_offset = seq_len * int(re.findall("\d+", freq)[0])
        # take care if more than two classes
        if self.binarize_target:
            if nb_classes == 2:
                df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
            else:
                df['y_pred'] = pd.qcut(df['y_pred'], nb_classes, labels=range(nb_classes))

        # split train val test
        df.reset_index(inplace=True)
        if df['datetime'].dtype == 'object':
            df['datetime'] = pd.to_datetime(df['datetime'])
        train = df.loc[df.datetime < train_end]
        train_datetime = train['datetime']
        train_x = train.copy()
        for col in colunns_to_drop:
            try:
                train_x.drop(columns=[col], inplace=True)
            except:
                pass
        train_y = train['y_pred']
        if self.time_step:
            train_time_step = train['time_step']
        else:
            train_time_step = None
        # validation
        val = df.loc[(df.datetime > (train_end - datetime.timedelta(minutes=minutes_offset)))
                    & (df.datetime < val_end)]
        val.reset_index(inplace=True)
        val_datetime = val['datetime']
        val_x = val.copy()
        for col in colunns_to_drop:
            try:
                val_x.drop(columns=[col], inplace=True)
            except:
                pass
        val_y = val['y_pred']
        if self.time_step:
            val_time_step = val['time_step']
        else:
            val_time_step = None
        # test
        test = df.loc[(df.datetime > (val_end - datetime.timedelta(minutes=minutes_offset))) &
                    (df.datetime < test_end)]
        test.reset_index(inplace=True)
        test_datetime = test['datetime']
        test_x = test.copy()
        for col in colunns_to_drop:
            try:
                test_x.drop(columns=[col], inplace=True)
            except:
                pass
        test_y = test['y_pred']
        if self.time_step:
            test_time_step = test['time_step']
        else:
            test_time_step = None
        # scaler has to be fit on train set, it's easier to do it here
        try:
            train_x = train_x.drop(columns=['time_step'])
            val_x = val_x.drop(columns=['time_step'])
            test_x = test_x.drop(columns=['time_step'])
            logger.info('time step succesfully droped')
        except:
            logger.info('didnt find time step in the dataset')
        if not sc_x:
            sc_x = StandardScaler()
            train_x = sc_x.fit_transform(train_x)
        else:
            train_x = sc_x.transform(train_x)
        val_x = sc_x.transform(val_x)
        test_x = sc_x.transform(test_x)
        joblib.dump(sc_x, os.path.join(config.package_directory, 'models',
                                    'sc_x_{}_{}.save'.format(config.currency, freq)))

        self.X, self.y = self.split_sequences(train_x, train_y, seq_len, train_datetime,
                            steps_ahead=steps_ahead, name='train', time_step=train_time_step)
        self.X_val, self.y_val = self.split_sequences(val_x, val_y, seq_len, val_datetime,
                                    steps_ahead=steps_ahead, name='val', time_step=val_time_step)
        self.X_test, self.y_test = self.split_sequences(test_x, test_y, seq_len, test_datetime,
                                        steps_ahead=steps_ahead, name='test', time_step=test_time_step)

        # You always have to give a 4D array as input to the cnn when using conv2d
        # So input data has a shape of (batch_size, height, width, depth)
        # if using conv2d instead of conv1d then:
        if config.input_dim == '2d':
            self.X = self.X.reshape(self.X.shape + (1,))
            self.X_val = self.X_val.reshape(self.X_val.shape + (1,))
            self.X_test = self.X_test.reshape(self.X_test.shape + (1,))

        self.y_cat = to_categorical(self.y)
        self.y_val_cat = to_categorical(self.y_val)
        self.y_test_cat = to_categorical(self.y_test)
        self.train = train

        logger.info(f'Shape of X train: {self.X.shape}')
        logger.info(f'Shape of X val: {self.X_val.shape}')
        logger.info(f'Shape of X test: {self.X_test.shape}')
        return None

    
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
        self.y_test = self.set_y_true(self.test_dataset)
        y_pred = self.model.predict(self.test_dataset)
        y_pred_class = [1 if y > 0.5 else 0 for y in y_pred]
        roc_auc = roc_auc_score(self.y_test, y_pred_class)
        f1 = f1_score(self.y_test, y_pred_class)
        logger.info('roc_auc : %.4f',roc_auc)
        logger.info('f1 : %.4f', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.log_metric('f1', f1)

        df = self.add_cost(self.df)
        df.reset_index(inplace=True)
        start_date, end_date = self.load_test_dates()

        lower_bounds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # lower_bounds = [0.5]
        upper_bounds = [1 - lower for lower in lower_bounds]
        # TODO: add compute_profitability_classes won't work with new tf.Dataset framework
        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            portfolio_result, hit_ratio, time_active = self.compute_profitability_classes(df, y_pred, 
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


    # #
    # upper_bound = 0.5
    # lower_bound = 0.5
    # date_start = start_date
    # date_end = end_date

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




# from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
# from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
# crypto = CryptoFactory(config.provider, config.currency)
# crypto.time_aggregation(freq=config.freq)
# crypto.run_3_barriers()
# df = crypto.df_3_barriers
# df = prepare_processed_dataset(df=df, add_target=False)
# logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
# df = df.merge(crypto.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
# logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
# experiment = Experiment(df, time_step=True, binarize_target=False)
# # df = crypto.df_time_aggregated
# # df = prepare_processed_dataset(df=df, add_target=True)
# # experiment = Experiment(df)
# experiment.train_test_val_split()
# experiment.train_model()
# experiment.evaluate_model()