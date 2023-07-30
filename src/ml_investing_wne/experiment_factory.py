import logging

from ml_investing_wne import config
from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
from ml_investing_wne.experiment import Experiment


logger = logging.getLogger(__name__)

def create_asset():

    if config.RUN_TYPE == 'forex':
        # TODO: something is worng with the order
        df = get_hist_data(currency=config.currency)
        asset = CryptoFactory(config.provider, config.currency, df=df)
    else:
        asset = CryptoFactory(config.provider, config.currency)

    return asset


class experiment_factory():

    def __init__(self, asset) -> None:
        self.asset = asset

    def get_experiment(self):

        if config.RUN_TYPE == 'forex':
            if config.provider == 'hist_data':
                experiment = self.forex_hist_data()
            else:
                logger.info('Flow not implemented!')
        elif config.RUN_TYPE == 'crypto':
            if config.RUN_SUBTYPE == 'time_aggregated':
                experiment = self.crypto_time_aggregated()
            elif config.RUN_SUBTYPE == 'volume_bars':
                experiment = self.crypto_volume_bars()
            elif config.RUN_SUBTYPE == 'triple_barrier_time_aggregated':
                experiment = self.crypto_triple_barrier_time_aggregated()
            else:
                logger.info('Flow not implemented!')
        else:
            logger.info('Flow not implemented!')
        return experiment

    def forex_hist_data(self):

        self.asset.run_3_barriers(t_final=config.t_final, fixed_barrier=config.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, time_step=config.time_step, binarize_target=False, asset_factory=self.asset)
        return experiment

    def crypto_time_aggregated(self):
        
        if self.asset.df_time_aggregated is None:
            self.asset.time_aggregation(freq=config.freq)
        df = self.asset.df_time_aggregated
        df = prepare_processed_dataset(df=df, add_target=True)
        experiment = Experiment(df)
        return experiment

    def crypto_volume_bars(self):
        
        if self.asset.df_volume_bars is None:
            self.asset.generate_volumebars(frequency=config.volume)
        df = self.asset.df_volume_bars
        df = prepare_processed_dataset(df=df, add_target=True)
        experiment = Experiment(df)
        return experiment
    

    def crypto_triple_barrier_time_aggregated(self):

        if self.asset.df_time_aggregated is None:
            self.asset.time_aggregation(freq=config.freq)
        self.asset.run_3_barriers(t_final=config.t_final, fixed_barrier=config.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, binarize_target=False, asset_factory=self.asset)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment

