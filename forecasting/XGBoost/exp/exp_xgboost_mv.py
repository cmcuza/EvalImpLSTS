from copy import deepcopy
from exp_basic import ExpBasic
import pandas as pd
import numpy as np
from itertools import product
from forecasting.XGBoost.components.xgboost_model import XGBoost
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from data.data_loader import Solar
from utils.metrics import metrics, MSE
from os.path import join
import os


class ExpXGBoostMV(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.parameters = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'subsample': args.subsample,
            # 'min_child_weight': args.min_child_weight
        }

        self.hyperparameters_path = os.path.join('output', 'xgboost', args.dataset, 'raw')

    def set_best_hyperparameter(self, index):
        n_estimators = [80, 100, 120]
        max_depth = [4, 6, 8]
        subsample = [0.8, 1]
        # min_child_weight = [0, 0.05, 0.1]
        ne, md, sp = list(product(n_estimators, max_depth, subsample))[index]
        self.parameters['n_estimators'] = ne
        self.parameters['max_depth'] = md
        self.parameters['subsample'] = sp
        # self.parameters['min_child_weight'] = mcw
        print('Best hyperparameters are', self.parameters)

    def change_hyperparameters(self):
        n_estimators = [80, 100, 120]
        max_depth = [4, 6, 8]
        subsample = [0.8, 1]
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
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        train_loader = Solar(root_path='./data/compressed/pmc/', data='solar_output_data_points.parquet')
        train_data = train_loader.data_x

        val_loader = Solar(root_path='./data/compressed/pmc/', data='solar_output_data_points.parquet', flag='val')
        val_data = val_loader.data_x

        self.model = list()
        for j in range(train_data.shape[1]):
            min_error = np.inf
            min_hyper = 0

            if os.path.exists(os.path.join(self.hyperparameters_path, f'hyperparameter_index_{j}.txt')):
                min_hyper = int(open(os.path.join(self.hyperparameters_path, f'hyperparameter_index_{j}.txt'), 'r').readline())
            else:
                for i in self.change_hyperparameters():
                    print("Training combination", i)
                    self.model_name = f'xgboost_v{j}'
                    model = self._build_model()
                    model.train(train_data[:, j])

                    true, pred = model.predict(val_data[:, j])
                    error = MSE(true, pred)
                    if error < min_error:
                        print("Error reduced from", round(min_error, 4), "to", round(error, 4), "with parameters",
                              self.parameters)
                        min_error = error
                        min_hyper = i
                open(os.path.join(self.hyperparameters_path, f'hyperparameter_index_{j}.txt'), 'w').write('%d' % min_hyper)

            self.set_best_hyperparameter(min_hyper)
            model = self._build_model()
            model.train(train_data[:, j])
            self.model.append(model)

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")

        test_loader = Solar(root_path='./data/compressed/pmc/', data='solar_output_data_points.parquet', flag='test')
        test_data = test_loader.data_x

        if not self.model:
            self.find_hyperparameters(data, model_name)

        for i in range(test_data.shape[1]):
            true, pred = self.model[i].predict(test_data[:, i])
            raw_file_root = join('output', 'xgboost', self.args.dataset, 'raw')
            os.makedirs(raw_file_root, exist_ok=True)
            with open(raw_file_root + f'/true_v{i}', 'wb') as f:
                pkl.dump(true, f)
            with open(raw_file_root + f'/output_v{i}', 'wb') as f:
                pkl.dump(pred, f)

            cm = metrics(pred, true)
            print(f'Results in raw v{i}', cm)

        self.run_ps_exp(model_name)

    def run_ps_exp(self, model_name):
        file_root = join('output', 'xgboost', self.args.dataset, self.args.eblc)

        for eb in self.args.EB:
            if eb == 0:
                continue
            print('Predicting with epsilon=', eb)

            test_loader = Solar(root_path=f'./data/compressed/{self.args.eblc}/',
                                data='solar_output_data_points.parquet',
                                eb=eb,
                                flag='test')

            test_data = test_loader.data_x

            for i in range(test_data.shape[1]):
                print('test size', test_data.shape)
                p, t = self.model[i].predict(test_data[:, i])

                prediction_path = join(file_root, 'predictions', model_name + f'eb_{eb}_v{i}_output.pkl')
                with open(prediction_path, 'wb') as f:
                    pkl.dump(p, f)

                print("Computing metrics")

                metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
                results = dict(zip(metrics_name, metrics(p, t)))

                print(f"Results of v{i}", results)

