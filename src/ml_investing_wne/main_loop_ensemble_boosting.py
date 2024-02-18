import argparse
import datetime
import os
import pickle
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, PredefinedSplit,
                                     RandomizedSearchCV)

from ml_investing_wne import config
from ml_investing_wne.experiment_factory import (create_asset,
                                                 experiment_factory)
from ml_investing_wne.performance_evaluator import PerformanceEvaluator
from ml_investing_wne.utils import get_logger

# if parameters won't be passed from command line, they will be taken from config.py
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


train_end = [datetime.datetime(2022, 1, 1, 0, 0, 0), datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0)]
val_end = [datetime.datetime(2022, 4, 1, 0, 0, 0), datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0)]
test_end = [datetime.datetime(2022, 7, 1, 0, 0, 0), datetime.datetime(2022, 10, 1, 0, 0, 0), datetime.datetime(2023, 1, 1, 0, 0, 0), datetime.datetime(2023, 4, 1, 0, 0, 0), datetime.datetime(2023, 7, 1, 0, 0, 0),]

logger = get_logger()
args.model = 'xgboost'
# number of experiments in grid search
n_experiments = 100
seeds = [12345, 123456, 1234567]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    config.seed = seed

def xgboost_tuning_random(train_val, ps, n_experiments=10, top_n=3):
    '''
    
    returns a dict of top n hyperparameters
    n_experiments: number of random experiments
    ps: PredefinedSplit object
    top_n: number of top results to return

    '''

    y_train_val = train_val['y_pred_bin']
    X_train_val = train_val.drop(['y_pred', 'y_pred_bin'], axis=1)

    # Define the hyperparameters to tune
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.2],
        'reg_lambda': [0.0, 0.1, 0.2]
    }

    # Create a RandomizedSearchCV object
    cv = RandomizedSearchCV(xgb.XGBClassifier(), param_distributions, scoring='accuracy', n_iter=n_experiments, cv=ps, n_jobs=-1, refit=False)

    # Fit the model to the training data
    cv.fit(X_train_val, y_train_val)

    results = pd.DataFrame(cv.cv_results_)
    results = results.sort_values(by='rank_test_score')
    results = results[:top_n]
    return results.to_dict('records')


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
                asset = create_asset(args)
                experiment = experiment_factory(asset, args).get_experiment()
                # in the standard experiment binarization happens in train test split. We need to keep y_pred continous for financial evaluation                         
                experiment.df['y_pred_bin'] = [1 if y > 1 else 0 for y in experiment.df['y_pred']]
                
                train_df = experiment.df[experiment.df.index < args.train_end]
                # remove last target length rows from train
                if experiment.offset > 0:
                    train_df = train_df.iloc[:-experiment.offset]
                val_df = experiment.df[(experiment.df.index >= args.train_end) & (experiment.df.index < args.val_end)]
                train_val_df = experiment.df[experiment.df.index < args.val_end]
                if experiment.offset > 0:
                    val_df = val_df.iloc[:-experiment.offset]
                    train_val_df = train_val_df.iloc[:-experiment.offset]
                test_df = experiment.df[(experiment.df.index >= args.val_end) & (experiment.df.index < args.test_end)]
                train_indices = np.where(train_df.index < args.train_end, -1, 0)
                val_indices = np.where(val_df.index < args.val_end, 0, 1)

                split_indices = np.append(train_indices, val_indices)
                ps = PredefinedSplit(test_fold=split_indices)
                # find best hyperparameters once and save into pickle
                if i == 0 and j==0 and m == 0:
                    top_models = xgboost_tuning_random(train_val_df, n_experiments=n_experiments, ps=ps)
                    dir_path = experiment.dir_path
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    with open(os.path.join(dir_path,'top_models.pickle'), 'wb') as handle:
                        pickle.dump(top_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(top_models[m]['mean_test_score'])
                xgb_model = xgb.XGBClassifier(objective='binary:logistic', **top_models[m]['params'])
                xgb_model.fit(train_val_df.drop(['y_pred','y_pred_bin'], axis=1), train_val_df['y_pred_bin'])
                
                y_pred = xgb_model.predict_proba(test_df.drop(['y_pred','y_pred_bin'], axis=1))
                test_df[f'prediction_{m}'] = y_pred[:,1]
                logger.info(f'Analyzing result for test period corresponding to {args.val_end.strftime("%Y%m%d")} - {args.test_end.strftime("%Y%m%d")}')
                # experiment.evaluate_model()
                if m == 0:
                    df_predictions = test_df
                else:  
                    df_predictions = df_predictions.join(test_df[f'prediction_{m}'])
                if m == 2:
                    df_predictions['prediction'] = df_predictions[['prediction_0','prediction_1', 'prediction_2']].mean(axis=1)
                    df_predictions['cost'] = args.cost
                    df_predictions.reset_index(inplace=True)
                    if 'triple_barrier' in experiment.args.run_subtype:
                        df_predictions = df_predictions.merge(experiment.df_3_barriers_additional_info, on='datetime', how='inner')
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
    daily_records = os.path.join(args.processed_data_path, f'binance_{args.currency}', 'time_aggregated_1440min.csv')
    performance_evaluator = PerformanceEvaluator(experiment.dir_path, daily_records, cost=args.cost)
    performance_evaluator.run()
        

if __name__ == "__main__":
    main()
 