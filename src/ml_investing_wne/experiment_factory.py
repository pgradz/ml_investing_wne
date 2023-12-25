import logging

from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
from ml_investing_wne.experiment import Experiment


logger = logging.getLogger(__name__)

def create_asset(args):

    if args.run_type == 'forex':
        # TODO: something is worng with the order
        df = get_hist_data(currency=args.currency)
        asset = CryptoFactory(args.provider, args.currency, args.run_subtype, df=df)
    else:
        asset = CryptoFactory(args.provider, args.currency, args.run_subtype)

    return asset


class experiment_factory():

    def __init__(self, asset, args) -> None:
        self.asset = asset
        self.args = args

    def get_experiment(self,  **kwargs):

        if self.args.run_type == 'forex':
            if self.args.provider == 'hist_data':
                experiment = self.forex_hist_data()
            else:
                logger.info('Flow not implemented!')
        elif self.args.run_type == 'crypto':
            if self.args.run_subtype == 'time_aggregated':
                experiment = self.crypto_time_aggregated(**kwargs)
            elif self.args.run_subtype == 'volume_bars':
                experiment = self.crypto_volume_bars(**kwargs)
            elif self.args.run_subtype == 'dollar_bars':
                experiment = self.crypto_dollar_bars(**kwargs)    
            elif self.args.run_subtype == 'triple_barrier_time_aggregated':
                experiment = self.crypto_triple_barrier_time_aggregated(**kwargs)
            elif self.args.run_subtype == 'volume_bars_triple_barrier':
                experiment = self.crypto_triple_barrier_volume_bars(**kwargs)
            elif self.args.run_subtype == 'dollar_bars_triple_barrier':
                experiment = self.crypto_triple_barrier_dollar_bars(**kwargs)
            elif self.args.run_subtype == 'cumsum':
                experiment = self.crypto_cumsum(**kwargs)
            elif self.args.run_subtype == 'cumsum_triple_barrier':
                experiment = self.crypto_cumsum_triple_barrier(**kwargs)
            elif self.args.run_subtype == 'range_bar':
                experiment = self.crypto_range_bar(**kwargs)
            elif self.args.run_subtype == 'range_bar_triple_barrier':
                experiment = self.crypto_triple_barrier_range_bar(**kwargs)
            else:
                logger.info('Flow not implemented!')
        else:
            logger.info('Flow not implemented!')
        return experiment

    def forex_hist_data(self, **kwargs):

        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, time_step=self.args.time_step, binarize_target=False, **kwargs)
        return experiment

    def crypto_time_aggregated(self, **kwargs):
        
        if self.asset.df_time_aggregated is None:
            self.asset.time_aggregation(freq=self.args.freq)
        df = self.asset.df_time_aggregated
        df = prepare_processed_dataset(df=df, add_target=True)
        experiment = Experiment(df, args=self.args, **kwargs)
        return experiment

    def crypto_volume_bars(self, **kwargs):
        
        if self.asset.df_volume_bars is None:
            self.asset.generate_volumebars(frequency=self.args.volume)
        df = self.asset.df_volume_bars
        df = prepare_processed_dataset(df=df, add_target=True)
        experiment = Experiment(df, args=self.args, **kwargs)
        return experiment

    def crypto_dollar_bars(self, **kwargs):
        
        if self.asset.df_dollar_bars is None:
            self.asset.generate_dollarbars(frequency=self.args.volume)
        df = self.asset.df_dollar_bars
        df = prepare_processed_dataset(df=df, add_target=True)
        experiment = Experiment(df, args=self.args, **kwargs)
        return experiment
    

    def crypto_triple_barrier_time_aggregated(self, **kwargs):

        # if self.asset.df_time_aggregated is None:
        #     self.asset.time_aggregation(freq=args.freq)
        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, args=self.args, binarize_target=False, **kwargs)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment
    
    def crypto_triple_barrier_volume_bars(self, **kwargs):
            
        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, args=self.args, binarize_target=False, **kwargs)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment

    def crypto_triple_barrier_dollar_bars(self, **kwargs):
            
        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, args=self.args, binarize_target=False, **kwargs)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment


    def crypto_cumsum(self, **kwargs):
        df = prepare_processed_dataset(df=self.asset.df, add_target=True)
        experiment = Experiment(df, args=self.args, **kwargs)
        return experiment
    
    def crypto_range_bar(self, **kwargs):
        df = prepare_processed_dataset(df=self.asset.df, add_target=True)
        experiment = Experiment(df, args=self.args,  **kwargs)
        return experiment

    
    def crypto_cumsum_triple_barrier(self, **kwargs):

        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, args=self.args, binarize_target=False, **kwargs)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment
    
    def crypto_triple_barrier_range_bar(self, **kwargs):
            
        self.asset.run_3_barriers(t_final=self.args.t_final, fixed_barrier=self.args.fixed_barrier)
        df = self.asset.df_3_barriers
        df = prepare_processed_dataset(df=df, add_target=False)
        logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
        # df = df.merge(self.asset.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
        logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        experiment = Experiment(df, args=self.args, binarize_target=False, **kwargs)
        experiment.df_3_barriers_additional_info = self.asset.df_3_barriers_additional_info
        return experiment



