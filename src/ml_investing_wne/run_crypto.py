from ml_investing_wne import config
from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
from ml_investing_wne.data_engineering.prepare_dataset import \
    prepare_processed_dataset

crypto = CryptoFactory(config.provider, config.currency)
crypto.generate_volumebars(frequency=4022)
crypto.plot_volumebars()
crypto.time_aggregation(freq='60min')

print(crypto.df.head())
print(crypto.df_volume_bars.head())
print(crypto.df_volume_bars.tail())
print(crypto.df_time_aggregated.head())

df = prepare_processed_dataset(df=crypto.df_time_aggregated)
