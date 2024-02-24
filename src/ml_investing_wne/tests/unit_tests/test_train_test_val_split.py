# import datetime

# import numpy as np
# import pandas as pd
# import pytest
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.utils import to_categorical

# from ml_investing_wne.train_test_val_split import (split_sequences,
#                                                    train_test_val_split)


# @pytest.fixture
# def train_x():
#     train_x = np.array([[1, 10, 100], [2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104],
#                         [6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108], [10, 19, 109]])
#     return train_x

# @pytest.fixture
# def train_y():
#     train_y = pd.Series(np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1]))
#     return train_y

# @pytest.fixture
# def train_datetime():
#     train_datetime = pd.date_range(datetime.datetime.now(), periods=10).tolist()
#     return train_datetime


# def test_train_test_val_split(nb_classes=2, freq='720min',
#                              seq_len=5, steps_ahead=1,
#                              train_end=datetime.datetime(2021, 1, 4, 12, 0, 0),
#                              val_end=datetime.datetime(2021, 1, 5, 12, 0, 0),
#                              test_end=datetime.datetime(2021, 1, 6, 12, 0, 0)):

#     datetime_col = [datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
#                     datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
#                     datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
#                     datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
#                     datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0)]

#     df_input = pd.DataFrame({"datetime": datetime_col, "Feature_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                              "Feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 120],
#                              "y_pred": [0, 0, 0, 0, 0, 1.1, 0, 0, 1.1, 1.1]})

#     sc_x = StandardScaler()
#     train_x = sc_x.fit_transform(df_input.iloc[0:6, 1:3])
#     train_y = df_input.iloc[4:6, 3].copy()
#     train_datetime = df_input.iloc[0:6, 0]
#     val = df_input.iloc[2:8,:].copy().reset_index()
#     val.drop(columns=['index'], inplace=True)
#     val_x = sc_x.transform(val.iloc[:, 1:3])
#     val_y = val.iloc[-2:, 3]
#     val_datetime = val.iloc[:, 0]
#     test = df_input.iloc[4:,:].copy().reset_index()
#     test.drop(columns=['index'], inplace=True)
#     test_x = sc_x.transform(test.iloc[:, 1:3])
#     test_y = test.iloc[-2:, 3]
#     test_datetime = test.iloc[:, 0]

#     X_output, y_output = split_sequences(train_x, train_y, seq_len, train_datetime,
#                            steps_ahead=steps_ahead, name='train')
#     X_val_output, y_val_output = split_sequences(val_x, val_y, seq_len, val_datetime,
#                                    steps_ahead=steps_ahead, name='val')
#     X_test_output, y_test_output = split_sequences(test_x, test_y, seq_len, test_datetime,
#                                      steps_ahead=steps_ahead, name='test')

#     y_cat_output = to_categorical(y_output)
#     y_val_cat_output = to_categorical(y_val_output)
#     y_test_cat_output = to_categorical(y_test_output)
#     df_input.set_index('datetime', inplace=True)
#     X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(
#         df_input, sc_x=None, nb_classes=nb_classes, freq=freq,seq_len=seq_len,
#         steps_ahead=steps_ahead, train_end=train_end, val_end=val_end, test_end=test_end)

#     assert X.shape[0] == X_output.shape[0]
#     assert X.shape[1] == X_output.shape[1]
#     assert X.shape[2] == X_output.shape[2]
#     assert y_cat.shape[0] == y_cat_output.shape[0]
#     assert y_cat.shape[1] == y_cat_output.shape[1]

#     assert X_val.shape[0] == X_val_output.shape[0]
#     assert X_val.shape[1] == X_val_output.shape[1]
#     assert X_val.shape[2] == X_val_output.shape[2]
#     assert y_val_cat.shape[0] == y_val_cat_output.shape[0]
#     assert y_val_cat.shape[1] == y_val_cat_output.shape[1]

#     assert X_test.shape[0] == X_test_output.shape[0]
#     assert X_test.shape[1] == X_test_output.shape[1]
#     assert X_test.shape[2] == X_test_output.shape[2]
#     assert y_test_cat.shape[0] == y_test_cat_output.shape[0]
#     assert y_test_cat.shape[1] == y_test_cat_output.shape[1]

#     for i in range(X.shape[0]):
#         np.testing.assert_array_equal(X[i], X_output[i])

#     for i in range(X_val.shape[0]):
#         np.testing.assert_array_equal(X_val[i], X_val_output[i])

#     for i in range(X_test.shape[0]):
#         np.testing.assert_array_equal(X_test[i], X_test_output[i])


# def test_split_sequences(train_x, train_y, train_datetime,
#                          steps_ahead=1, seq_len=5):
#     output_array_1 = np.array(
#         [[1, 10, 100], [2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104]])
#     output_array_2 = np.array(
#         [[2, 11, 101], [3, 12, 102], [4, 13, 103], [5, 14, 104], [6, 15, 105]])
#     output_array_3 = np.array(
#         [[3, 12, 102], [4, 13, 103], [5, 14, 104], [6, 15, 105], [7, 16, 106]])
#     output_array_4 = np.array(
#         [[4, 13, 103], [5, 14, 104], [6, 15, 105], [7, 16, 106], [8, 17, 107]])
#     output_array_5 = np.array(
#         [[5, 14, 104], [6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108]])
#     output_array_6 = np.array(
#         [[6, 15, 105], [7, 16, 106], [8, 17, 107], [9, 18, 108], [10, 19, 109]])
#     output_X = np.array([output_array_1, output_array_2, output_array_3, output_array_4,
#                          output_array_5, output_array_6])
#     output_y = [0, 1, 0, 0, 1, 1]
#     X, y = split_sequences(train_x, train_y, seq_len, train_datetime,
#                            steps_ahead=steps_ahead, name='train')
#     assert X.shape[0] == output_X.shape[0]
#     assert X.shape[1] == output_X.shape[1]
#     assert X.shape[2] == output_X.shape[2]

#     for i in range(X.shape[0]):
#         np.testing.assert_array_equal(X[i], output_X[i])

#     assert len(y) == len(output_y)
#     assert all([a == b for a, b in zip(y, output_y)])
