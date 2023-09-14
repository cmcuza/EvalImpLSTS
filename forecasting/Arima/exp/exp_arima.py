from copy import deepcopy
from exp_basic import ExpBasic
import pandas as pd
import os
from pmdarima import auto_arima
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils.metrics import metrics
from os.path import join, exists
from tqdm import tqdm


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
        best_k = 0
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
                                   n_jobs=8,
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

        # return best_model.arima_res_, k - 1
        #x_test = scaler.transform(test)
        #p, t = self.get_predictions(deepcopy(best_model.arima_res_), x_test, self.get_harmonics(test, K=best_k))
        #prediction_path = join(file_root, 'output.pkl')

        #with open(prediction_path, 'wb') as f:
        #    pkl.dump(p, f)

        #true_path = join(file_root, 'true.pkl')
        #with open(true_path, 'wb') as f:
        #   pkl.dump(t, f)

        #r = metrics(p, t)
        #print('Baseline results', r)

        return best_model, best_k

    def get_predictions(self, model, data, harmonics):
        predictions = list()
        true = list()
        data = np.squeeze(data)
        for i in tqdm(range(0, len(data) - self.pred_len + 1 - self.pred_len, self.pred_len)):
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
        raw_columns = [f'{self.args.target_var}-R', 'datetime']

        train_data, val_data, test_data = self.temporal_train_val_test_split(full_dataset[raw_columns].copy())
        train_data = pd.concat([train_data, val_data])
        if not (self.model or self.k):
            self.model, self.k = self.get_best_arima(train_data, test_data)

        train, test = train_data.set_index('datetime'), test_data.set_index('datetime')
        file_root = join('output', 'arima', self.args.dataset, self.args.eblc)
        scaler = StandardScaler()
        scaler.fit(train)

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
            compressed_test_data.rename({f'{self.args.target_var}-E{eb}': f'{self.args.target_var}-R'}, axis=1, inplace=True)
            print('test size', compressed_test_data.shape)

            model_result = deepcopy(self.model.arima_res_)

            x_test = scaler.transform(compressed_test_data)
            p, t = self.get_predictions(model_result, x_test, self.get_harmonics(test, K=self.k))

            prediction_path = join(file_root, 'predictions', model_name + f'eb_{eb}_output.pkl')
            with open(prediction_path, 'wb') as f:
                pkl.dump(p, f)

            print("Computing metrics")

            metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
            results = dict(zip(metrics_name, metrics(p, t)))

            print("Results ", results)





