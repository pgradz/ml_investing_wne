import unittest
from unittest.mock import patch
import datetime
import pandas as pd
import numpy as np
from ml_investing_wne.experiment import Experiment
import pandas.testing as pd_testing


class TestExperiment(unittest.TestCase):

    def setUp(self) -> None:
        # first two records will be discarded when running crypto_factory.get_daily_volatility. Last three rows will be discarded because of t_final=3
        self.train_x = np.array([[1, 10, 100], [2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104],
                        [6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108], [10, 19, 109]])
        self.train_y = pd.Series(np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1]))

        self.datetime_col = [datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
                    datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
                    datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                    datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
                    datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0)]

        self.df_input = pd.DataFrame({"datetime": self.datetime_col, "Feature_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                             "Feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 120],
                             "y_pred": [0, 0, 0, 0, 0, 1.1, 0, 0, 1.1, 1.1]})
        
        self.time_step =  pd.Series([2,1,3,1,2,3,2,2,1,1])

        self.experiment = Experiment(df=self.df_input, binarize_target=True, time_step=None, asset_factory=None)
                
    def test_split_sequences(self):

        output_array_1 = np.array(
            [[1, 10, 100], [2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104]])
        output_array_2 = np.array(
            [[2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104], [6, 15, 105]])
        output_array_3 = np.array(
            [[3, 12, 102], [4, 13, 103], [5, 14, 104], [6, 15, 105], [7, 16, 106]])
        output_array_4 = np.array(
            [[4, 13, 103], [5, 14, 104], [6, 15, 105], [7, 16, 106], [8, 17, 107]])
        output_array_5 = np.array(
            [[5, 14, 104], [6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108]])
        output_array_6 = np.array(
            [[6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108], [10, 19, 109]])
        output_X = np.array([output_array_1, output_array_2, output_array_3, output_array_4,
                            output_array_5, output_array_6])
        output_y = [0, 1, 0, 0, 1, 1]
        train_datetime = pd.Series(self.datetime_col)

        X, y = self.experiment.split_sequences(self.train_x, self.train_y, n_steps=5, datetime_series=train_datetime,
                            steps_ahead=1, name='train')

        assert X.shape[0] == output_X.shape[0]
        assert X.shape[1] == output_X.shape[1]
        assert X.shape[2] == output_X.shape[2]

        for i in range(X.shape[0]):
            np.testing.assert_array_equal(X[i], output_X[i])

        assert len(y) == len(output_y)
        assert all([a == b for a, b in zip(y, output_y)])


    def test_split_sequences_time_step(self):

        output_array_1 = np.array(
            [[1, 10, 100], [2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104]])
        output_array_3 = np.array(
            [[3, 12, 102], [4, 13, 103], [5, 14, 104], [6, 15, 105], [7, 16, 106]])
        output_array_5 = np.array(
            [[5, 14, 104], [6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108]])
        output_array_6 = np.array(
            [[6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108], [10, 19, 109]])
        output_X = np.array([output_array_1, output_array_3,
                            output_array_5, output_array_6])
        output_y = [0, 0, 1, 1]
        train_datetime = pd.Series(self.datetime_col)

        X, y = self.experiment.split_sequences(self.train_x, self.train_y, n_steps=5, datetime_series=train_datetime,
                            steps_ahead=1, name='train', time_step=self.time_step)

        assert X.shape[0] == output_X.shape[0]
        assert X.shape[1] == output_X.shape[1]
        assert X.shape[2] == output_X.shape[2]

        for i in range(X.shape[0]):
            np.testing.assert_array_equal(X[i], output_X[i])

        assert len(y) == len(output_y)
        assert all([a == b for a, b in zip(y, output_y)])

if __name__ == '__main__':
    unittest.main()



