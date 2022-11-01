import logging
import pandas as pd

from ml_investing_wne import config


logger = logging.getLogger(__name__)

def get_units():
    units = {
        'EURCHF': 100000,
        'EURPLN': 10000
    }
    return units

def prepare_xtb_data(resp):

    units = get_units()
    try:
        unit = units[config.currency]
    except KeyError:
        logger.error('Please check conversion for this symbol and add it to units')
        
    df = pd.DataFrame(resp['returnData']['rateInfos'])
    df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
    df['close'] = (df['open'] + df['close'])/unit
    df['high'] = (df['open'] + df['high'])/unit
    df['low'] = (df['open'] + df['low'])/unit
    df['open'] = df['open']/unit
    df['datetime'] = df['datetime'].dt.tz_localize('GMT').dt.tz_convert('Europe/Warsaw').dt.tz_localize(None)
    df = df.set_index('datetime')
    df.drop(columns=['ctm', 'ctmString', 'vol'], inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    df = df.resample(config.freq).agg({'open': 'first',
                                                'high': 'max',
                                                'low': 'min',
                                                'close': 'last'
                                                })

    df.dropna(inplace=True)
    df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close']
    df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
    df['datetime'] = df.index
    df['hour'] = df.datetime.dt.hour

    return df


