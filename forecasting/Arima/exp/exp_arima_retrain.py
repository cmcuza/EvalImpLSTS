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


class ExpArimaRetrain(ExpBasic):
    def __init__(self, args):
        super().__init__(args)

        for eblc in ['pmc', 'swing', 'sz']:
            os.makedirs(join(self.args.output_root, 'arima_retrain', args.dataset, eblc, 'predictions'), exist_ok=True)
            os.makedirs(join(self.args.output_root, 'arima_retrain', args.dataset, eblc, 'models'), exist_ok=True)
            os.makedirs(join(self.args.output_root, 'arima_retrain', args.dataset, eblc, 'metrics'), exist_ok=True)

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

    def get_best_arima(self, train, eb, eblc):
        train = train.set_index('datetime')
        K = 3
        best_aic = np.inf
        best_model = None
        best_k = 0
        file_root = join('output', 'arima_retrain', self.args.dataset, 'raw')
        os.makedirs(file_root, exist_ok=True)
        scaler = StandardScaler()
        x_train = np.squeeze(scaler.fit_transform(train))

        for k in range(1, K):
            print(f'Testing harmonic {k}')
            train_exog = self.get_harmonics(train, K=k)

            model_path = join(file_root, f'arima_{eb}_harmonic_{k}_{eblc}.pkl')
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

        return best_model, best_k, scaler

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
        print("Running retraining", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        full_dataset = pd.read_parquet(data)
        if 'sz' in data:
            full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])
        else:
            full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'], unit='ms')

        raw_columns = [f'{self.args.target_var}-R', 'datetime']

        train_data, val_data, test_data = self.temporal_train_val_test_split(full_dataset[raw_columns].copy())
        train_data = pd.concat([train_data, val_data])
        model, k, scaler = self.get_best_arima(train_data, 'raw', self.args.eblc)
        test = test_data.set_index('datetime')
        model_result = model.arima_res_

        x_test = scaler.transform(test)
        p, t = self.get_predictions(model_result, x_test, self.get_harmonics(test, K=k))
        file_root = join('output', 'arima_retrain', self.args.dataset, 'raw')
        prediction_path = join(file_root, 'output.pkl')
        with open(prediction_path, 'wb') as f:
            pkl.dump(p, f)

        file_root = join('output', 'arima_retrain', self.args.dataset, self.args.eblc)

        # for eb in self.args.EB:
        #     if eb == 0:
        #         continue
        #
        #     if data.find('sz') != -1:
        #         eb = eb * 0.01
        #
        #     if data.find('aus') != -1:
        #         eb = float(eb)
        #
        #     eb_error_columns = ['datetime', f'{self.args.target_var}-E{eb}']
        #
        #     temp_full_dataset = full_dataset[eb_error_columns].copy()
        #     print('Retraining on', self.args.eblc, eb, 'with size', temp_full_dataset.shape)
        #     train_data, val_data, test_data = self.temporal_train_val_test_split(temp_full_dataset)
        #     train_data = pd.concat([train_data, val_data])
        #     model, k, scaler = self.get_best_arima(train_data, eb, self.args.eblc)
        #     test_data = test_data.set_index('datetime')
        #     print(test_data.head())
        #     print('test size', test_data.shape)
        #
        #     model_result = model.arima_res_
        #
        #     x_test = scaler.transform(test_data)
        #     p, t = self.get_predictions(model_result, x_test, self.get_harmonics(test, K=k))
        #
        #     prediction_path = join(file_root, 'predictions', model_name + f'eb_{eb}_output.pkl')
        #     with open(prediction_path, 'wb') as f:
        #         pkl.dump(p, f)
        #
        #     print("Computing metrics")
        #
        #     metrics_name = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr']
        #     results = dict(zip(metrics_name, metrics(p, t)))
        #
        #     print("Results ", results)





