import random
import numpy as np
import tensorflow as tf
import mlflow.keras
import datetime

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory



train_end = [datetime.datetime(2022, 1, 1, 0, 0, 0), datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0)]
val_end = [datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0)]
test_end = [datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0), datetime.datetime(2023, 7, 1, 0, 0, 0)]

seeds = [12345, 123456, 1234567]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    config.seed = seed

logger = get_logger()

logger.info('Tensorflow has access to the following devices:')
for device in tf.config.list_physical_devices():
    logger.info(f'{device}')


def main():

    for i, seed in enumerate(seeds):
        set_seed(seed)
        logger.info(f'Processing {i}  of {len(seeds)} experiments with seed {seed}')
        # initalize budget and reset it for each larger experiment
        trading_result = 100
        hit_counter = 0
        trades = 0
        for train, val, test in zip(train_end, val_end, test_end):
            # just in case set seed once again, it used to reset after each experiment
            set_seed(seed)
            config.train_end = train
            config.val_end = val
            config.test_end = test
            config.seed = seed
            mlflow.tensorflow.autolog()
            asset = create_asset()
            experiment = experiment_factory(asset).get_experiment(train_end=config.train_end, 
                                                        val_end=config.val_end,
                                                        test_end=config.test_end, seed=config.seed)
            experiment.train_test_val_split()
            experiment.hyperparameter_tunning()
            model = experiment.get_model()
            # once again
            set_seed(seed)
            experiment.set_budget(trading_result)
            experiment.set_model(model)
            if i == 0:
                experiment.model.summary()
            experiment.train_model()
            logger.info(f'Analyzing result for test period corresponding to {config.val_end.strftime("%Y%m%d")} - {config.test_end.strftime("%Y%m%d")}')
            experiment.evaluate_model()
            trading_result = experiment.get_budget()
            hit_counter += experiment.get_hit_counter()
            trades += experiment.get_trades_counter()

        logger.info(f'Final budget: {trading_result}')
        logger.info(f'Final hit ratio: {hit_counter/trades}')    
        

if __name__ == "__main__":
    main()
