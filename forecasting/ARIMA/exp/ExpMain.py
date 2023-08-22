from exp_basic import ExpBasic
import pandas as pd
import os
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils.metrics import metrics
from utils.plotting import print_forecast
from os.path import join, exists


class ExpMain(ExpBasic):

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

    def get_best_arima(self, train, test, model_name, ace=0.0):
        train, test = train.set_index('datetime'), test.set_index('datetime')
        K = 25
        best_aic = np.inf
        best_model = None
        k = 0
        for k in range(1, K):
            print(f'Testing harmonic {k}')
            train_exog = self.get_harmonics(train, K=k)
            scaler = StandardScaler()
            x_train = np.squeeze(scaler.fit_transform(train))

            file_root = join('output', 'models')
            file_path = join(file_root, model_name+f'_harmonic_{k}.pkl')
            if not exists(file_path):

                model = auto_arima(x_train,
                                   X=train_exog.values,
                                   start_p=0,
                                   start_q=0,
                                   trace=True,
                                   n_jobs=5,
                                   seasonal=False,
                                   error_action='warn',
                                   supress_warnings=True,
                                   stepwise=False,
                                   n_fits=50,
                                   random_state=42)
                print(model.summary())
                os.makedirs(file_root, exist_ok=True)
                with open(file_path, 'wb') as f:
                    pkl.dump(model, f)
            else:
                with open(file_path, 'rb') as f:
                    model = pkl.load(f)

            if best_aic > model.aic():
                print('Improved AIC from', best_aic, 'to', model.aic())
                best_aic = model.aic()
                best_model = model
            else:
                break

        if k == 24:
            k = 25

        print('Best arima with harmonic K =', k - 1)

        if ace != 0.0:
            return best_model.arima_res_, k - 1

        p = self.get_prediction_results(best_model.arima_res_, test,
                                        self.get_harmonics(test, K=k - 1))

        # print_forecast(test, p, f'{model_name}_ace_0.0',
        #                {'name': 'forecast_arima', 'data': model_name})

        r = metrics(test, p)
        # r = np.ones(3)
        print('MAE, MSE, MAPE', r)
        return best_model, r, k - 1

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

    def get_prediction_results(self, model, data, harmonics):
        predictions = np.zeros((len(data) - self.pred_len + 1, self.pred_len))
        for i in range(0, len(data) - self.pred_len + 1):
            predictions[i] = model.forecast(self.pred_len, exog=harmonics.iloc[i:i + self.pred_len])
            model = model.append([data[i]], exog=harmonics.loc[[harmonics.index[i]], :])

        return predictions

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        full_dataset = pd.read_parquet(data)
        # full_dataset = full_dataset[:10000]  # delete
        full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])
        columns = full_dataset.columns
        raw_columns = list(filter(lambda c: c[-1] == 'R', columns))
        raw_columns = ['datetime'] + raw_columns
        temp_full_dataset = full_dataset[raw_columns].copy()
        train_data, val_data, test_data = self.temporal_train_val_test_split(temp_full_dataset, eb='R')

        best_model, r, k = self.get_best_arima(train_data, test_data, model_name)

        print('Did this')



