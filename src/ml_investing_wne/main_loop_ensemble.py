import argparse
import os
import random
import numpy as np
import tensorflow as tf
import mlflow.keras
import datetime

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory
from ml_investing_wne.performance_evaluator import PerformanceEvaluator


parser = argparse.ArgumentParser(description='Most commonly changing settings')

parser.add_argument('--run_type', type=str, default=config.run_type, help='data sampling and target labelling method')
parser.add_argument('--run_subtype', type=str, default=config.run_subtype, help='data sampling and target labelling method')
parser.add_argument('--currency', type=str, default=config.currency, help='currency to be trained on')
parser.add_argument('--model', type=str, default=config.model, help='model to be used')
parser.add_argument('--freq', type=str, default=config.freq, help='frequency for time aggegation')
parser.add_argument('--volume', type=int, default=config.volume, help='volume for volume bars')
parser.add_argument('--value', type=int, default=config.value, help='value for dollar bars')
parser.add_argument('--t_final', type=int, default=config.t_final, help='vertical barrier for triple barrier method')
parser.add_argument('--fixed_barrier', type=float, default=config.fixed_barrier, help='bottom and upper barrier for triple barrier method')
parser.add_argument('--cumsum_threshold', type=float, default=config.cumsum_threshold, help='bottom and upper barrier for triple barrier method')

parser.add_argument('--seq_stride', type=int, default=config.seq_stride, help='allows to skip consequtive sequences')
parser.add_argument('--seq_len', type=int, default=config.seq_len, help='input sequence length')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
parser.add_argument('--patience', type=int, default=config.patience, help='patience for early stopping')
parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
parser.add_argument('--nb_classes', type=int, default=config.nb_classes, help='number of classes')
parser.add_argument('--steps_ahead', type=int, default=config.steps_ahead, help='number of steps ahead for prediction')

parser.add_argument('--cost', type=float, default=config.cost, help='cost of transaction')
parser.add_argument('--package_directory', type=str, default=config.package_directory, help='path to package directory')
parser.add_argument('--model_path_final', type=str, default=config.model_path_final, help='path to save final model')
parser.add_argument('--processed_data_path', type=str, default=config.processed_data_path, help='path to processed data')
parser.add_argument('--model_path', type=str, default=config.model_path, help='path to save model')
parser.add_argument('--provider', type=str, default=config.provider, help='data provider')

args = parser.parse_args()

train_end = [datetime.datetime(2022, 1, 1, 0, 0, 0), datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0),
              datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0)]
val_end = [datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), 
           datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0)]
test_end = [datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0),
             datetime.datetime(2023, 4, 1, 0, 0, 0), datetime.datetime(2023, 7, 1, 0, 0, 0)]

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
                args.train_end = dates[0]
                args.val_end = max(dates[1], test_start_date_next_interval)
                args.test_end = dates[2]
                args.seed = seed
                mlflow.tensorflow.autolog()
                asset = create_asset(args)
                experiment = experiment_factory(asset, args).get_experiment()
                experiment.train_test_val_split()
                # make sure this is only done once - it is solved like that because of problem with keras tuner and tf on macos
                if i == 0 and j ==0 and m == 0: 
                    models, best_hps, my_hyper_model= experiment.hyperparameter_tunning(m)
                # model = copy.deepcopy(models[m])
                model = my_hyper_model.tuner.hypermodel.build(best_hps[m])
                # once again
                set_seed(seed)
                experiment.set_budget(trading_result)
                experiment.set_model(model)
                experiment.model.summary()
                experiment.train_model()
                logger.info(f'Analyzing result for test period corresponding to {args.val_end.strftime("%Y%m%d")} - {args.test_end.strftime("%Y%m%d")}')
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
        try:
            logger.info(f'Final hit ratio: {hit_counter/trades}')
        except ZeroDivisionError:
            logger.info(f'Final hit ratio: no trades made')
           
        
    # summarize results from different seeds
    # daily end prices are needed for some performance metrics
    daily_records = os.path.join(args.processed_data_path, f'binance_{args.currency}', 'time_aggregated_1440min.csv')
    performance_evaluator = PerformanceEvaluator(experiment.dir_path, daily_records, cost=args.cost)
    performance_evaluator.run()

if __name__ == "__main__":
    main()
 