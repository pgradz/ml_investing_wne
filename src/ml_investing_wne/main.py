import random

import mlflow.keras
import numpy as np
import tensorflow as tf

from ml_investing_wne import config
from ml_investing_wne.experiment_factory import (create_asset,
                                                 experiment_factory)
from ml_investing_wne.utils import get_logger

random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

logger = get_logger()

logger.info('Tensorflow has access to the following devices:')
for device in tf.config.list_physical_devices():
    logger.info(f'{device}')

def main():
    # autlog ends run after keras.fit, hence all logging later will be added to another run. This is
    # inconvenient, but the alternative is to manually create run and replicate functionalities of
    # autlog

    mlflow.tensorflow.autolog()
    asset = create_asset()
    experiment = experiment_factory(asset).get_experiment()
    experiment.run()

if __name__ == "__main__":
    main()
