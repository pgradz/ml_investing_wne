import datetime
import unittest
from unittest.mock import patch, MagicMock
from unittest import mock
from io import StringIO

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import tensorflow as tf

from ml_investing_wne import config
from ml_investing_wne.experiment import Experiment


class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:

        self.mock_args = mock.MagicMock()
        self.mock_args.exchange = 'Binance'
        self.mock_args.currency = 'BTCUSDT'
        self.mock_args.seq_len = 2
        self.mock_args.run_subtype = 'TEST_triple_barrier'
        self.mock_args.model = 'LSTM'
        self.mock_args.train_end = datetime.datetime(2021,1,10) 
        self.mock_args.val_end = datetime.datetime(2021,1,15)
        self.mock_args.test_end = datetime.datetime(2021,1,21)
        self.mock_args.t_final =  2
        self.mock_args.seq_stride = 1
        self.mock_args.batch_size = 2
        self.mock_args.nb_classes = 2

        datetime_col = [datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
                            datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
                            datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                            datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
                            datetime.datetime(2021, 1, 5, 12, 0, 0)
                            ]

        _open = [28149.84, 28623.77, 29495.09, 30199.45, 28855.16, 29806.07, 30264.27, 30264.27,30264.27]
        close = [28623.76, 28228.25, 30199.44, 30466.58, 28612, 30264.27, 30264.27, 30264.27, 31172]
        high = [28644.96, 28819.71, 30308.23, 30550, 29237.29, 30485, 30264.37, 30264.37,31192]
        low = [28038.83, 28210.14,29429.77, 29800, 28520, 29806.07, 30264.17, 30264.17,31172]
        prc_change = [0.05, 0.05,  -0.05, -0.05, 0.05,  (31172/30264.27 -1), 0.05, 0.05, 0.05]
        barrier_touched = ['top', 'top', 'bottom','bottom', 'top', 'vertical', 'top','top', 'top']
        barrier_touched_date = [datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 2, 12, 0, 0),datetime.datetime(2021, 1, 3, 12, 0, 0),datetime.datetime(2021, 1, 3, 12, 0, 0),
                                datetime.datetime(2021, 1, 4, 0, 0, 0), datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),datetime.datetime(2021, 1, 6, 12, 0, 0),
                                datetime.datetime(2021, 1, 9, 0, 0, 0)]
        time_step = [2, 1, 2, 1, 1, 3, 3, 3, 3]

        self.df = pd.DataFrame({"datetime": datetime_col,
                            "open": _open,
                            "high": high,
                            "low": low,
                            "close": close,
                            "prc_change": prc_change,
                            "barrier_touched": barrier_touched,
                            "barrier_touched_date": barrier_touched_date,
                            "time_step": time_step
                            })
        self.df['bottom_barrier'] = self.df['close'] * 0.95
        self.df['top_barrier'] = self.df['close'] * 1.05
        self.df['cost'] = 0.001

        self.df_2 = pd.DataFrame({
            'y_pred': [0, 0, 1.2, 0, 0, 1.1, 0, 0, 1.1, 1.1, 0.9, 1.2, 0.95, 1.05, 1.1, 1.1, 1.11, 1.14, 1.15, 0.8],
            'datetime': pd.date_range(start='2021-01-01', periods=20),
            'index': range(20),
            # 'to_keep': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Feature_1': range(20),
            'Feature_2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 22]
        })
        self.df_2.set_index('datetime', inplace=True)
        self.experiment = Experiment(df=self.df, args=self.mock_args)
        self.experiment_2 = Experiment(df=self.df_2, args=self.mock_args)
        

    @patch("pandas.DataFrame.to_csv")
    def test_correct_predictions_triple_barrier(self, to_csv):
        to_csv.return_value = True
        self.df['prediction'] = [0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8]
        self.experiment.backtest(self.df, 0.4, 0.6)
        self.assertEqual(self.experiment.hit_counter, 5)
        self.assertEqual(self.experiment.trades_counter, 5)
        self.assertAlmostEqual(self.experiment.budget, 123.95, places=2)


    @patch('ml_investing_wne.experiment.StandardScaler')
    @patch('ml_investing_wne.experiment.joblib.dump')
    def test_train_test_val_split(self, mock_dump, mock_scaler):

        mock_scaler.return_value.transform.side_effect = lambda x: x  # Mock the transform method to return the input
        mock_scaler.return_value.fit_transform.side_effect = lambda x: x  # Mock the fit_transform method to return the input

        self.experiment_2.train_test_val_split()
        # get first batch from self.experiment_2.train_dataset
        for batch in self.experiment_2.train_dataset.take(1):
            X, y = batch
        # convert X and y to np.array
        X = X.numpy() 
        y = y.numpy()
        expected_X = np.array([[[0, 10], [1, 20]], [[1, 20], [2, 30]]])
        expected_y = np.array([[1, 0], [0, 1]])
        # compare X and expected_X
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

     
        expected_train_index = pd.DataFrame({'index': range(1,7),
                                             'datetime': pd.date_range(start='2021-01-02', periods=6)
                                                })
        np.testing.assert_array_equal(self.experiment_2.train_date_index, expected_train_index)

        expected_test_index = pd.DataFrame({'index': range(14,20),
                                             'datetime': pd.date_range(start='2021-01-15', periods=6)
                                                })
        np.testing.assert_array_equal(self.experiment_2.test_date_index, expected_test_index)


        # Check if the datasets are created correctly
        self.assertIsInstance(self.experiment_2.train_dataset, tf.data.Dataset)
        self.assertIsInstance(self.experiment_2.val_dataset, tf.data.Dataset)
        self.assertIsInstance(self.experiment_2.test_dataset, tf.data.Dataset)

    def mock_predict(self, input_data):
        if input_data is self.experiment_2.val_dataset:
            return np.array([[0.1,0.9], [0.1,0.9], [0.9,0.1]])
        elif input_data is self.experiment_2.test_dataset:
            return np.array([[0.1,0.9], [0.1,0.9], [0.9,0.1], [0.1,0.9], [0.1,0.9],[0.9,0.1]])
        else:
            raise ValueError("Unknown dataset")

    @patch('ml_investing_wne.experiment.Experiment.backtest')
    @patch('ml_investing_wne.experiment.StandardScaler')
    @patch('ml_investing_wne.experiment.joblib.dump')
    @patch("ml_investing_wne.experiment.logger")
    @patch("ml_investing_wne.experiment.mlflow")
    @patch('ml_investing_wne.experiment.Experiment.set_model')
    def test_evaluate_model(self, mock_set_model, mock_mlflow, mock_logger, mock_dump, mock_scaler, mock_backtest):
        mock_scaler.return_value.transform.side_effect = lambda x: x
        mock_scaler.return_value.fit_transform.side_effect = lambda x: x

        self.experiment_2.train_test_val_split()
        mock_model = MagicMock()
        self.experiment_2.model = mock_model
        self.experiment_2.model.predict = MagicMock(side_effect=self.mock_predict)
        self.experiment_2.df_3_barriers_additional_info = pd.DataFrame({'datetime': pd.date_range(start='2021-01-01', periods=20),
                                                                        'barrier_touched': ['top', 'vertical', 'bottom', 'bottom', 'top',
                                                                                            'top', 'vertical', 'bottom', 'bottom', 'top',
                                                                                            'top', 'vertical', 'bottom', 'bottom', 'top',
                                                                                            'top', 'vertical', 'bottom', 'bottom', 'top']
                                                                        })
                                                                                                                                                                             
        # Mock the evaluate function of the TensorFlow model
        mock_model.evaluate.return_value = [0.25, 0.95]
        mock_backtest.return_value = True
        self.experiment_2.evaluate_model()
    
        expected_df_eval_test = StringIO('''idx,y_pred,index,Feature_1,Feature_2,datetime,train_val_test,prediction,cost,barrier_touched
                                        14,1.1,14.0,14.0,170.0,2021-01-15,test,0.9,0.0,top
                                        15,1.1,15.0,15.0,180.0,2021-01-16,test,0.9,0.0,top
                                        16,1.11,16.0,16.0,190.0,2021-01-17,test,0.1,0.0,vertical
                                        17,1.14,17.0,17.0,200.0,2021-01-18,test,0.9,0.0,bottom
                                        18,1.15,18.0,18.0,210.0,2021-01-19,test,0.9,0.0,bottom
                                        19,0.8,19.0,19.0,22.0,2021-01-20,test,0.1,0.0,top
                                    ''')

        expected_df_eval_test = pd.read_table(expected_df_eval_test, sep=',', parse_dates=['datetime'],index_col=0)
        expected_df_eval_test.index.name = None
        pd_testing.assert_frame_equal(self.experiment_2.df_eval_test, expected_df_eval_test)


      
if __name__ == '__main__':
    unittest.main()


