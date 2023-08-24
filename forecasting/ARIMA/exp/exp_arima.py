import time

from exp_basic import ExpBasic
import pandas as pd
import os
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils.metrics import metrics
from os.path import join, exists


class ExpArima(ExpBasic):
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

    def get_best_arima(self, train, test):
        train, test = train.set_index('datetime'), test.set_index('datetime')
        K = 3
        best_aic = np.inf
        best_model = None
        k = 0
        file_root = join('output', 'arima', self.args.dataset, 'raw')
        os.makedirs(file_root, exist_ok=True)
        scaler = StandardScaler()
        x_train = np.squeeze(scaler.fit_transform(train))

        for k in range(1, K):
            print(f'Testing harmonic {k}')
            train_exog = self.get_harmonics(train, K=k)

            model_path = join(file_root, f'arima_raw_harmonic_{k}.pkl')
            if not exists(model_path):

                model = auto_arima(x_train,
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
            else:
                break

        print('Best arima with harmonic K =', k)

        # return best_model.arima_res_, k - 1
        x_test = scaler.transform(test)
        p, t = self.get_predictions(best_model.arima_res_, x_test, self.get_harmonics(test, K=k))
        prediction_path = join(file_root, 'output.pkl')

        with open(prediction_path, 'wb') as f:
            pkl.dump(p, f)

        true_path = join(file_root, 'true.pkl')
        with open(true_path, 'wb') as f:
            pkl.dump(t, f)

        r = metrics(p, t)
        print('Baseline results', r)

        return best_model, k

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
        # data = data[:1000]
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
        full_dataset = pd.read_parquet(data)
        full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])
        columns = full_dataset.columns
        raw_columns = list(filter(lambda c: c[-1] == 'R', columns))
        raw_columns = ['datetime'] + raw_columns
        # temp_full_dataset = full_dataset[raw_columns].copy()
        train_data, val_data, test_data = self.temporal_train_val_test_split(full_dataset[raw_columns].copy(), eb='R')

        if not (self.model or self.k):
            self.model, self.k = self.get_best_arima(train_data, test_data)

        train, test = train_data.set_index('datetime'), test_data.set_index('datetime')
        file_root = join('output', 'arima', self.args.dataset, self.args.eblc)
        scaler = StandardScaler()
        x_train = np.squeeze(scaler.fit_transform(train))

        train_exog = self.get_harmonics(train, K=self.k)

        for eb in self.args.EB:
            print('Predicting with epsilon=', eb)

            if eb != 0:
                if data.find('sz') != -1:
                    eb = eb * 0.01

                if data.find('aus') != -1:
                    eb = float(eb)

                eb_error_columns = list(filter(lambda c: c.split('-')[-1] == f'E{eb}', columns))

                eb_error_columns.append('datetime')
                temp_full_dataset = full_dataset[eb_error_columns].copy()
                _, _, compressed_test_data = self.temporal_train_val_test_split(temp_full_dataset, f'E{eb}')
                compressed_test_data = compressed_test_data.set_index('datetime')
            else:
                continue

            print('test size', compressed_test_data.shape)

            # model_path = join(file_root, 'models', model_name + f'eb_{eb}.pkl')
            model = ARIMA(endog=x_train, exog=train_exog.values, order=self.model.order)
            model_result = model.fit()

            # with open(model_path, 'wb') as f:
            #     pkl.dump(model_result, f)

            x_test = scaler.transform(compressed_test_data)
            p, t = self.get_predictions(model_result, x_test, self.get_harmonics(test, K=self.k))

            prediction_path = join(file_root, 'predictions', model_name + f'eb_{eb}_output.pkl')
            with open(prediction_path, 'wb') as f:
                pkl.dump(p, f)

            # true_path = join(file_root, 'predictions', model_name + f'eb_{eb}_true.pkl')
            # with open(true_path, 'wb') as f:
            #     pkl.dump(t, f)

            print("Computing metrics")

            metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
            results = dict(zip(metrics_name, metrics(p, t)))

            print("Results ", results)





