import random
import numpy as np
import tensorflow as tf
import mlflow.keras

from ml_investing_wne import config
from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import get_ml_flow_experiment_name, get_callbacks, \
    get_final_model_path, evaluate_model
from ml_investing_wne.models import model_factory
from ml_investing_wne.utils import get_logger
from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.experiment import Experiment
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

logger = get_logger()


def main():
    # autlog ends run after keras.fit, hence all logging later will be added to another run. This is
    # inconvenient, but the alternative is to manually create run and replicate functionalities of
    # autlog

    mlflow.tensorflow.autolog()
    if config.RUN_TYPE == 'forex':
        if config.provider == 'hist_data':
            df = get_hist_data(currency=config.currency)
            crypto = CryptoFactory(config.provider, config.currency, df=df)
            df = prepare_processed_dataset(df=df, add_target=False)
            crypto.run_3_barriers()
            df = crypto.df_3_barriers
            logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
            df = df.merge(crypto.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
            logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
        else:
            logger.error('not implemented')
    elif config.RUN_TYPE == 'crypto':
        crypto = CryptoFactory(config.provider, config.currency)
        if config.RUN_SUBTYPE == 'time_aggregated':
            crypto.time_aggregation(freq=config.freq)
            df = crypto.df_time_aggregated
            df = prepare_processed_dataset(df=df, add_target=True)
            experiment = Experiment(df)
        elif config.RUN_SUBTYPE == 'volume_bars':
            crypto.generate_volumebars(frequency=config.volume)
            df = crypto.df_volume_bars
            df = prepare_processed_dataset(df=df, add_target=True)
            experiment = Experiment(df)
        elif config.RUN_SUBTYPE == 'triple_barrier_time_aggregated':
            crypto.time_aggregation(freq=config.freq)
            crypto.run_3_barriers(t_final=10)
            df = crypto.df_3_barriers
            df = prepare_processed_dataset(df=df, add_target=False)
            logger.info(f' df shape before merge wiith 3 barriers additional info is {df.shape}')
            df = df.merge(crypto.df_3_barriers_additional_info[['datetime', 'time_step']], on='datetime', how='inner')
            logger.info(f' df shape after merge wiith 3 barriers additional info is {df.shape}')
            experiment = Experiment(df, time_step=True, binarize_target=False, asset_factory=crypto)
        else:
            logger.info('Flow not implemented!')

    experiment.run()
    # X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, _ = train_test_val_split(df, 
    #                      nb_classes=config.nb_classes, freq=config.freq,
    #                      seq_len=config.seq_len, steps_ahead=config.steps_ahead,
    #                      train_end=config.train_end, val_end=config.val_end,
    #                      test_end=config.test_end, binarize_target=binarize_target, time_step=time_step)
    # mlflow.set_experiment(experiment_name=get_ml_flow_experiment_name())
    # callbacks = get_callbacks()
    # model = model_factory(X)
    # history = model.fit(X, y_cat, batch_size=config.batch, epochs=config.epochs, verbose=2,
    #                     validation_data=(X_val, y_val_cat), callbacks=callbacks)
    # model.save(get_final_model_path())
    # evaluate_model(model, df, X_test, y_test_cat, y, y_val, y_test)

if __name__ == "__main__":
    main()
