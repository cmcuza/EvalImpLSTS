from exp_basic import ExpBasic
import pandas as pd
import os
from pmdarima import auto_arima
from data.data_loader import Solar
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from copy import deepcopy
import pickle as pkl
from utils.metrics import metrics
from os.path import join, exists


class ExpArimaMV(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.k = None

    def get_harmonics(self, ts, freq='15T', K=1):
        harmonics = pd.DataFrame({'date': ts.index})
        harmonics['date'] = pd.PeriodIndex(harmonics['date'], freq=freq)
        harmonics.set_index('date', inplace=True)
        harmonics.sort_index(inplace=True)
        for k in range(1, K + 1):
            harmonics[f'sin-{k}har'] = np.sin(
                k * 2 * np.pi * (harmonics.index.hour * 60 + harmonics.index.minute) / (24 * 60))
            harmonics[f'cos-{k}har'] = np.cos(
                k * 2 * np.pi * (harmonics.index.hour * 60 + harmonics.index.minute) / (24 * 60))

        return harmonics

    def get_best_arima(self, train, train_stamp, test, test_stamp, ps):
        K = 3
        k_list = list()
        file_root = join('output', 'arima', self.args.dataset, 'raw')
        os.makedirs(file_root, exist_ok=True)
        best_models = list()
        for i in range(ps, ps+train.shape[1]):  # ps+train.shape[1]):
            best_k = 0
            best_model = None
            best_aic = np.inf
            for k in range(1, K):
                print(f'Testing harmonic {k}')
                train_exog = self.get_harmonics(train_stamp, K=k)

                model_path = join(file_root, f'arima_raw_harmonic_{k}_v{i}.pkl')
                if not exists(model_path):

                    model = auto_arima(train[:, i],
                                       X=train_exog.values,
                                       start_p=0,
                                       start_q=0,
                                       trace=True,
                                       n_jobs=10,
                                       seasonal=False,
                                       error_action='warn',
                                       supress_warnings=True,
                                       stepwise=False,
                                       n_fits=50,
                                       random_state=42)
                    print(model.summary())

                    with open(model_path, 'wb') as f:
                        pkl.dump(model, f)
                else:
                    with open(model_path, 'rb') as f:
                        model = pkl.load(f)

                if best_aic > model.aic():
                    print('Improved AIC from', best_aic, 'to', model.aic())
                    best_aic = model.aic()
                    best_model = model
                    best_k = k
                else:
                    break

            print('Best arima with harmonic K =', best_k)

            p, t = self.get_predictions(best_model.arima_res_, test[:, i],
                                        self.get_harmonics(test_stamp, K=best_k))
            prediction_path = join(file_root, f'output_v{i}.pkl')

            with open(prediction_path, 'wb') as f:
                pkl.dump(p, f)

            true_path = join(file_root, f'true_v{i}.pkl')
            with open(true_path, 'wb') as f:
                pkl.dump(t, f)

            r = metrics(p, t)
            print(f'Baseline results for variable {i} is', r)

            best_models.append(best_model)
            k_list.append(best_k)

        return best_models, k_list

    def train_model(self, data_loader, model_order, k, ace, exp_name):
        file_root = join('..', 'trained_models', data_loader.name, 'arima', exp_name)
        os.makedirs(file_root, exist_ok=True)
        file_path = join(file_root, f'{data_loader.name}_arima_harmonic_{k}_ace_{np.round(ace, 4)}.pkl')
        if not exists(file_path):
            train, test = data_loader.get_train_test_values()
            train_exog, _ = data_loader.get_harmonics(K=k)
            model = ARIMA(endog=train, exog=train_exog, order=model_order)
            res = model.fit()
            print(res.summary())

            with open(file_path, 'wb') as f:
                pkl.dump(res, f)
            return res
        else:
            with open(file_path, 'rb') as f:
                return pkl.load(f)

    def get_predictions(self, model, data, harmonics):
        predictions = list()
        true = list()
        data = np.squeeze(data)
        for i in range(0, len(data) - self.pred_len + 1 - self.pred_len, self.pred_len):
            predictions.append(model.forecast(self.pred_len, exog=harmonics.iloc[i:i + self.pred_len]))
            true.append(data[i:i + self.pred_len])
            model = model.append(data[i:i+self.pred_len], exog=harmonics.iloc[i:i+self.pred_len].values)

        return np.asarray(predictions), np.asarray(true)

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        train_loader = Solar(root_path='./data/compressed/pmc/', data='solar_output_data_points.parquet')
        train_data, train_timestamps = train_loader.data_x, train_loader.data_timestamp

        test_loader = Solar(root_path='./data/compressed/pmc/', data='solar_output_data_points.parquet', flag='test')
        test_data, test_timestamps = test_loader.data_x, test_loader.data_timestamp

        if not (self.model or self.k):
            self.model, self.k = self.get_best_arima(train_data, train_timestamps, test_data, test_timestamps, 0)

        self.run_ps_exp(model_name, train_data, 0)

    def run_ps_exp(self, model_name, raw_train_data, ps):
        file_root = join('output', 'arima', self.args.dataset, self.args.eblc)

        for eb in self.args.EB:
            if eb == 0:
                continue
            print('Predicting with epsilon=', eb)

            test_loader = Solar(root_path=f'./data/compressed/{self.args.eblc}/',
                                data='solar_output_data_points.parquet',
                                eb=eb,
                                flag='test')

            test_data, test_timestamps = test_loader.data_x, test_loader.data_timestamp

            for i in range(ps, ps + raw_train_data.shape[1]):
                print('test size', test_data.shape)

                # model_path = join(file_root, 'models', model_name + f'eb_{eb}.pkl')
                model = deepcopy(self.model[i])
                model_result = model.arima_res_

                p, t = self.get_predictions(model_result, test_data[:, i],
                                            self.get_harmonics(test_timestamps, K=self.k[i]))

                prediction_path = join(file_root, 'predictions', model_name + f'eb_{eb}_v{i}_output.pkl')
                with open(prediction_path, 'wb') as f:
                    pkl.dump(p, f)

                print("Computing metrics")

                metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
                results = dict(zip(metrics_name, metrics(p, t)))

                print("Results ", results)





