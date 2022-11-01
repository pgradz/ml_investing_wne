import os
import logging
import datetime
import time
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
import ml_investing_wne.config as config
import csv

# logger settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
file_h = logging.FileHandler(os.path.join(config.package_directory, 'logs', 'trading.log'))
stream_h.setLevel(logging.INFO)
file_h.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
stream_h.setFormatter(formatter)
file_h.setFormatter(formatter)
logger.addHandler(stream_h)
logger.addHandler(file_h)

start = datetime.datetime(2021, 10, 1, 1, 0, 0, 0)
symbol = 'EURCHF'

client = APIClient()

loginResponse = client.execute(loginCommand(userId=config.USER_ID, password=config.PASSWORD))
logger.info(str(loginResponse))

# check if user logged in correctly
if (loginResponse['status'] == False):
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']
balance = client.commandExecute('getMarginLevel')

while (True):
    row = {}
    tick = client.commandExecute('getTickPrices', {"level": 0,
                                                             "symbols": [symbol],
                                                             "timestamp": int(
                                                                 datetime.datetime.now().timestamp() * 1000 - 20000)
                                                             }
                                           )
    try:
        row['timestamp'] = tick['returnData']['quotations'][0]['timestamp']
        row['spread'] = tick['returnData']['quotations'][0]['spreadTable']
        row['ask'] = tick['returnData']['quotations'][0]['ask']
        row['bid'] = tick['returnData']['quotations'][0]['bid']
    except:
        pass


    with open(f'./spread_table_{symbol}.csv','a') as fd:
        w = csv.DictWriter(fd, row.keys())
        w.writerow(row)
    if datetime.datetime.now().hour not in [20, 21, 22, 23, 0, 1, 2, 11, 12, 13,
                                            14, 15, 16, 17, 18, 19, 4, 5, 6,
                                            7, 8, 9, 10]:
        break
    else:
        time.sleep(20)


