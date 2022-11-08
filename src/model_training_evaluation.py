import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from logger import define_logger
from config import Config
import optuna
from optuna.integration import LightGBMPruningCallback 
import json
import mlflow
import os


def objective_lgbm(trial, X:pd.DataFrame, y:pd.DataFrame, eval_metric:str):
    param_grid = {
        "verbose":-1,
        "random_state": Config.RANDOM_SEED,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 20, step=1),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

    cv_scores = np.empty(5)
    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        model = LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric= eval_metric,
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, eval_metric)
            ],  # Add a pruning callback
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = recall_score(y_eval, preds)

    return np.mean(cv_scores)



if __name__ == '__main__':

    # Create directory for features folder
    Config.FEATURE_PATH.mkdir(parents = True, exist_ok = True)

    # Logging
    logger = define_logger()
    logger.info('#### Model Training and Evaluation ####')
    X_train = pd.read_csv(Config.FEATURE_PATH / 'Train_features.csv')
    y_train = pd.read_csv(Config.FEATURE_PATH / 'Train_label.csv')
    X_test = pd.read_csv(Config.FEATURE_PATH / 'Test_features.csv')
    y_test = pd.read_csv(Config.FEATURE_PATH / 'Test_label.csv')

    # Convert object type columns to category
    for i in X_train.columns:
        if X_train[i].dtypes == 'O':
            X_train[i] = X_train[i].astype('category')
            X_test[i] = X_test[i].astype('category')

        # Reset the index
        X_train.reset_index(inplace = True, drop = True)
        y_train.reset_index(inplace = True, drop = True)

    # Parameters tuning
    study_lgbm = optuna.create_study(direction ='maximize', study_name='LGBM')
    func_lgbm = lambda trial: objective_lgbm(trial, X_train, y_train, 'auc')
    study_lgbm.optimize(func_lgbm, n_trials=30)

    #print(f"Best value: {study_lgbm.best_value:.5f}")

    parameters = {}
    for key, value in study_lgbm.best_params.items():
        parameters[key] = value

    # print(f"Best params:")
    # print(parameters)

    model = LGBMClassifier(**parameters)
    model.fit(X_train,y_train)
    pred_test = model.predict(X_test)

    acc_score = accuracy_score(y_test, pred_test)
    pre_score = precision_score(y_test, pred_test)
    reca_score = recall_score(y_test, pred_test)
    roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])


    # If the file exists, read the data.
    if os.path.exists(Config.METRICS_PATH):

        # Compare the history performance
        # Same as: with open ('./assets/metrics.json')
        with open(Config.METRICS_PATH, 'r') as output:

            data = json.load(output)
            avg_acc = 0
            avg_prec = 0
            avg_reca = 0
            avg_roc = 0
            for i in data:
                avg_acc = sum(d['accuracy'] for d in data.values())/len(data)
                avg_prec = sum(d['precision'] for d in data.values())/len(data)
                avg_reca = sum(d['recall'] for d in data.values())/len(data)
                avg_roc = sum(d['roc_auc'] for d in data.values())/len(data)
        
        # Model with better performance
        if acc_score > avg_acc or pre_score > avg_prec or reca_score > avg_reca or roc_score > avg_roc:
            if acc_score > avg_acc*0.8 and pre_score > avg_prec*0.8 and reca_score > avg_reca*0.8 and roc_score >avg_roc*0.8:
                
                logger.info('Better model detected')

                index = int(list(data)[-1][list(data)[-1].find('_')+1:])+1
                name = 'trail_' + str(index)
                data[name] = {'accuracy': acc_score, 'precision': pre_score, 
                                    'recall': reca_score, 'roc_auc': roc_score}

                # Logging
                logger.info('Better model detected at ' +name +'\n')

                # Store into next key in metrics.json
                with open(Config.METRICS_PATH, "w") as output2:
                    json.dump(data, output2, indent = 4)

                # Store the bestter model in mlflow
                with mlflow.start_run():

                    mlflow.log_param('Params', parameters)
                    mlflow.log_metric('Auc', roc_score)
                    mlflow.log_metric('Acc', acc_score)
                    mlflow.log_metric('Prec', pre_score)
                    mlflow.log_metric('Recall', reca_score)
                    mlflow.lightgbm.log_model(model, str(model)[:str(model).find('(')])

                    # mlflow.lightgbm.log_model(model, "LGBM")
                    # mlflow.sklearn.log_model, mlflow.xgboost.log_model
                    # print ui at cmd: mlflow ui

        else:
            logger.info('No improvement'+'\n')
    else:
        # If json file not found, create a new one with 'a+'
        with open(Config.METRICS_PATH, 'a+') as output3:
        
            parent_dict = {} 
            dictionary = {
                'accuracy': acc_score,
                'precision': pre_score,
                'recall': reca_score,
                'roc_auc': roc_score
                }

            parent_dict['trial_1'] = dictionary
            json.dump(parent_dict, output3) 
            logger.info('Registered first model' +'\n')

        # Store the bestter model in mlflow
        with mlflow.start_run():

            mlflow.log_param('Params', parameters)
            mlflow.log_metric('Auc', roc_score)
            mlflow.log_metric('Acc', acc_score)
            mlflow.log_metric('Prec', pre_score)
            mlflow.log_metric('Recall', reca_score)
            mlflow.lightgbm.log_model(model, str(model)[:str(model).find('(')])