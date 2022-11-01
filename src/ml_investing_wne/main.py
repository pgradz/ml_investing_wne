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

random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

logger = get_logger()

def main():
    # autlog ends run after keras.fit, hence all logging later will be added to another run. This is
    # inconvenient, but the alternative is to manually create run and replicate functionalities of
    # autlog
    mlflow.tensorflow.autolog()
    df = get_hist_data(currency=config.currency)
    df = prepare_processed_dataset(df=df)
    X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, _ = train_test_val_split(df)
    mlflow.set_experiment(experiment_name=get_ml_flow_experiment_name())
    callbacks = get_callbacks()
    model = model_factory(X)
    history = model.fit(X, y_cat, batch_size=config.batch, epochs=config.epochs, verbose=2,
                        validation_data=(X_val, y_val_cat), callbacks=callbacks)
    model.save(get_final_model_path())
    evaluate_model(model, df, X_test, y_test_cat, y, y_val, y_test)

if __name__ == "__main__":
    main()
