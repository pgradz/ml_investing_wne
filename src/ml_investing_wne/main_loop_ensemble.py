import os
import random
import numpy as np
import tensorflow as tf
import mlflow.keras
import datetime
import copy

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory
from ml_investing_wne.PerformanceEvaluator import PerformanceEvaluator


train_end = [datetime.datetime(2022, 1, 1, 0, 0, 0), datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0)]
val_end = [datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0)]
test_end = [datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0), datetime.datetime(2023, 7, 1, 0, 0, 0),]


# train_end = [datetime.datetime(2021, 10, 1, 0, 0, 0), datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0)]
# val_end = [datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0)]
# test_end = [datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0), datetime.datetime(2023, 7, 1, 0, 0, 0)]

# # train_end = [datetime.datetime(2023, 1, 1, 0, 0, 0)]
# val_end =  [datetime.datetime(2023, 4, 1, 0, 0, 0)]
# test_end = [datetime.datetime(2023, 7, 1, 0, 0, 0)]

seeds = [12345, 123456, 1234567]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    config.seed = seed

logger = get_logger()
logger.handlers.pop() 

logger.info('Tensorflow has access to the following devices:')
for device in tf.config.list_physical_devices():
    logger.info(f'{device}')


def main():

    for i, seed in enumerate(seeds):
        set_seed(seed)
        logger.info(f'Processing {i+1}  of {len(seeds)} experiments with seed {seed}')
        # initalize budget and reset it for each larger experiment
        trading_result = 100
        hit_counter = 0
        trades = 0
        # this is a placeholder to deal with situation when triple barrier method ends in next quarter
        test_start_date_next_interval = val_end[0]
        for j, dates in enumerate(zip(train_end, val_end, test_end)):
            
            # ensemble of 3 best models
            for m in range(3):
            # just in case set seed once again, it used to reset after each experiment
                set_seed(seed)
                config.train_end = dates[0]
                config.val_end = max(dates[1], test_start_date_next_interval)
                config.test_end = dates[2]
                config.seed = seed
                mlflow.tensorflow.autolog()
                asset = create_asset()
                experiment = experiment_factory(asset).get_experiment(train_end=config.train_end, 
                                                            val_end=config.val_end,
                                                            test_end=config.test_end, seed=config.seed)
                experiment.train_test_val_split()
                # make sure this is only done once
                if i == 0 and j ==0 and m == 0: 
                    models = experiment.hyperparameter_tunning(m)
                model = copy.deepcopy(models[m])
                # once again
                set_seed(seed)
                experiment.set_budget(trading_result)
                experiment.set_model(model)
                experiment.model.summary()
                experiment.train_model()
                logger.info(f'Analyzing result for test period corresponding to {config.val_end.strftime("%Y%m%d")} - {config.test_end.strftime("%Y%m%d")}')
                experiment.evaluate_model()
                if m == 0:
                    predictions = experiment.df_eval_test
                    predictions.rename(columns={'prediction': f'prediction_{m}'}, inplace=True)
                    df_predictions = predictions
                else:  
                    predictions = experiment.df_eval_test[['prediction']]
                    predictions.rename(columns={'prediction': f'prediction_{m}'}, inplace=True)
                    df_predictions = df_predictions.join(predictions)
                if m == 2:
                    df_predictions['prediction'] = df_predictions[['prediction_0','prediction_1', 'prediction_2']].mean(axis=1)
                    experiment.df_eval_test = df_predictions
                    experiment.set_budget(trading_result)
                    logger.info(f'Ensemble evaluation:')  
                    experiment.evaluate_model_short()

            trading_result = experiment.get_budget()
            hit_counter += experiment.get_hit_counter()
            trades += experiment.get_trades_counter()   
            try:
                logger.info(f'Running hit ratio: {hit_counter/trades}')   
            except ZeroDivisionError:
                logger.info(f'Running hit ratio: no trades yet')
            test_start_date_next_interval = experiment.test_start_date_next_interval

        logger.info(f'Final budget: {trading_result}')
        logger.info(f'Final hit ratio: {hit_counter/trades}')    
        
    # summarize results from different seeds
    # daily end prices are needed for some performance metrics
    daily_records = os.path.join(config.processed_data_path, f'binance_{config.currency}', 'time_aggregated_1440min.csv')
    performance_evaluator = PerformanceEvaluator(experiment.dir_path, daily_records )
    performance_evaluator.run()

if __name__ == "__main__":
    main()
 