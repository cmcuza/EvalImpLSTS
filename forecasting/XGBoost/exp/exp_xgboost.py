from copy import deepcopy
from exp_basic import ExpBasic
import pandas as pd
import numpy as np
from itertools import product
from forecasting.XGBoost.components.xgboost_model import XGBoost
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils.metrics import metrics, MSE
from os.path import join
import os


class ExpXGBoost(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.parameters = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'subsample': args.subsample,
            # 'min_child_weight': args.min_child_weight
        }

    def set_best_hyperparameter(self, index):
        n_estimators = [100, 120, 80]
        max_depth = [4, 6, 8]
        subsample = [0.5, 0.8, 1]
        # min_child_weight = [0, 0.05, 0.1]
        ne, md, sp = list(product(n_estimators, max_depth, subsample))[index]
        self.parameters['n_estimators'] = ne
        self.parameters['max_depth'] = md
        self.parameters['subsample'] = sp
        # self.parameters['min_child_weight'] = mcw
        print('Best hyperparameters are', self.parameters)

    def change_hyperparameters(self):
        n_estimators = [32, 64, 128]
        max_depth = [1, 2, 3]
        subsample = [0, 0.05, 0.1]
        # min_child_weight = [0, 0.05, 0.1]
        for i, (ne, md, sp) in enumerate(product(n_estimators, max_depth, subsample)):
            self.parameters['n_estimators'] = ne
            self.parameters['max_depth'] = md
            self.parameters['subsample'] = sp
            # self.parameters['min_child_weight'] = mcw
            yield i

    def _build_model(self):
        return XGBoost(self.seq_len, self.pred_len,
                       self.parameters['n_estimators'],
                       self.parameters['max_depth'],
                       self.parameters['subsample'])

    def find_hyperparameters(self, data, model_name):
        full_dataset = pd.read_parquet(data)
        best_model = None
        min_error = np.inf
        min_hyper = 0
        full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])
        # columns = full_dataset.columns
        # raw_columns = list(filter(lambda c: c[-1] == 'R', columns))
        raw_columns = ['datetime', self.args.target_var + '-R']

        train_data, val_data, _ = self.temporal_train_val_test_split(full_dataset[raw_columns].copy())

        train = train_data.set_index('datetime').values
        val = val_data.set_index('datetime').values
        # test = test_data.set_index('datetime')

        for i in self.change_hyperparameters():
            print("Training combination", i)
            self.model_name = model_name
            self.model = self._build_model()
            self.model.train(train)

            true, pred = self.model.predict(val)
            error = MSE(true, pred)
            if error < min_error:
                min_error = error
                min_hyper = i

        self.set_best_hyperparameter(min_hyper)

        self.model = None

        return best_model

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        full_dataset = pd.read_parquet(data) #  [:1000]
        full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])
        raw_columns = ['datetime', f'{self.args.target_var}-R']

        train_data, val_data, test_data = self.temporal_train_val_test_split(full_dataset[raw_columns].copy())
        train = train_data.set_index('datetime')
        test = test_data.set_index('datetime')
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        raw_test = scaler.transform(test)

        if not self.model:
            print('Training in raw data')
            self.model = self._build_model()
            self.model.train(train)
            true, pred = self.model.predict(raw_test)
            raw_file_root = join('output', 'xgboost', self.args.dataset, 'raw')
            os.makedirs(raw_file_root, exist_ok=True)
            with open(raw_file_root+'/true', 'wb') as f:
                pkl.dump(true, f)
            with open(raw_file_root+'/output', 'wb') as f:
                pkl.dump(pred, f)

            cm = metrics(pred, true)
            print('Results in raw', cm)

        file_root = join('output', 'xgboost', self.args.dataset, self.args.eblc)

        for eb in self.args.EB:
            if eb == 0:
                continue

            if data.find('sz') != -1:
                eb = eb * 0.01

            if data.find('aus') != -1:
                eb = float(eb)

            eb_error_columns = ['datetime', f'{self.args.target_var}-E{eb}']

            temp_full_dataset = full_dataset[eb_error_columns].copy()
            print('Predicting', self.args.eblc, eb, 'with size', temp_full_dataset.shape)
            _, _, compressed_test_data = self.temporal_train_val_test_split(temp_full_dataset)
            compressed_test_data = compressed_test_data.set_index('datetime')

            print('test size', compressed_test_data.shape)

            x_test = scaler.transform(compressed_test_data)
            true, pred = self.model.predict(x_test)

            prediction_path = join(file_root, 'predictions', model_name + f'_eb_{eb}_output.pkl')
            with open(prediction_path, 'wb') as f:
                pkl.dump(pred, f)

            print("Computing metrics")

            metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
            results = dict(zip(metrics_name, metrics(pred, true)))

            print("Results ", results)





