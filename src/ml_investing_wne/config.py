import os

package_directory = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(package_directory, 'data', 'raw')
raw_data_path_test = os.path.join(package_directory, 'data', 'test')
processed_data_path = os.path.join(package_directory, 'data', 'processed')

currencies = ['USDCHF']
