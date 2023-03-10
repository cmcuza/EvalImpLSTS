import os
from os.path import join
import torch
import pandas as pd
import pickle as pkl
from utils.metrics import *
from darts import TimeSeries
from data.data_factory import data_provider
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.parameters = dict()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.ot = args.target_var
        self.random_state = np.random.randint(1000)
        if self.args.data.find('ETT') != -1:
            self.fixed_borders = True
        else:
            self.fixed_borders = False

        self.model_name = ''
        self.device = self._acquire_device()

        if not os.path.isdir(self.args.output_root):
            os.makedirs(self.args.output_root)
            os.makedirs(join(self.args.output_root, 'predictions'))
            os.makedirs(join(self.args.output_root, 'models'))
            os.makedirs(join(self.args.output_root, 'metrics'))

    def _build_model(self):
        raise NotImplementedError

    def create_sequence(self, data):
        x_seq_data = []
        y_seq_data = []
        for i in range(data.shape[0] - self.seq_len - self.pred_len):
            train_seq = data.iloc[i:i + self.seq_len]
            train_label = data.iloc[i + self.seq_len:i + self.seq_len + self.pred_len]
            x_seq_data.append(TimeSeries.from_dataframe(train_seq, time_col='datetime'))
            y_seq_data.append(TimeSeries.from_dataframe(train_label, time_col='datetime'))

        return x_seq_data, y_seq_data

    def temporal_train_val_test_split(self, data, eb):

        if self.fixed_borders:
            # border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            # border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1s = [0, 6000 - self.seq_len, 8000 - self.seq_len]
            border2s = [6000, 8000, 10000]
        else:
            num_train = int(len(data) * 0.7)
            num_test = int(len(data) * 0.2)
            num_val = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_val, len(data)]

        train_border1 = border1s[0]
        train_border2 = border2s[0]
        target_column = ['datetime', self.ot + '-' + eb]

        train_data = data[target_column][train_border1:train_border2].reset_index(drop=True)

        val_border1 = border1s[1]
        val_border2 = border2s[1]

        val_data = data[target_column][val_border1:val_border2].reset_index(drop=True)

        test_border1 = border1s[2]
        test_border2 = border2s[2]

        test_data = data[target_column][test_border1:test_border2].reset_index(drop=True)

        return train_data, val_data, test_data

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def val(self, val_data, val_loader, criterion):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test):
        pass

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.seq_len, "and", self.pred_len)
        self.model_name = model_name
        print("Loading the data")
        full_dataset = pd.read_parquet(data)
        full_dataset = full_dataset[:10000]  # delete
        full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])

        columns = full_dataset.columns
        raw_columns = list(filter(lambda c: c[-1] == 'R', columns))
        raw_columns = ['datetime'] + raw_columns
        temp_full_dataset = full_dataset[raw_columns].copy()
        train_data, val_data, test_data = self.temporal_train_val_test_split(temp_full_dataset, eb='R')

        print('train size', train_data.shape)
        x_train = TimeSeries.from_dataframe(train_data, time_col='datetime')

        print('val size', val_data.shape)
        x_val = TimeSeries.from_dataframe(val_data, time_col='datetime')

        scaler = Scaler(StandardScaler())
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        model = self._build_model()

        print("Training")

        model.train(x_train, x_val)

        for eb in self.args.EB:
            print('Predicting with epsilon=', eb)
            if eb != 0:
                if data.find('sz') != -1:
                    eb = eb * 0.01
                eb_error_columns = list(filter(lambda c: c.split('-')[-1] == f'E{eb}', columns))

                eb_error_columns.append('datetime')
                temp_full_dataset = full_dataset[eb_error_columns].copy()
                _, _, all_test_data = self.temporal_train_val_test_split(temp_full_dataset, f'E{eb}')

            print('test size', test_data.shape)
            x_test = TimeSeries.from_dataframe(test_data, time_col='datetime')
            x_test = scaler.transform(x_test)

            raw_p_values = model.predict(x_test)

            seq_x_test, seq_y_test = self.create_sequence(test_data)

            print("Computing metrics")

            raw_y_values = np.asarray([scaler.transform(e).all_values() for e in seq_y_test]).squeeze()

            results = {
                "MAE": np.round(MAE(raw_y_values, raw_p_values), 4),
                "MSE": np.round(MSE(raw_y_values, raw_p_values), 4),
                "RMSE": np.round(RMSE(raw_y_values, raw_p_values), 4)
            }

            print("Results", results)

            print('Storing the results')

            new_model_name = "_".join([self.model_name]+['eb', str(eb)])

            pd.DataFrame(results, index=[0]).to_csv(join(self.args.output_root, 'metrics', f"testing_{new_model_name}.csv"), index=False)

            with open(join(self.args.output_root, "predictions", f"testing_{new_model_name}_output.pickle"), 'wb') as f:
                pkl.dump(raw_p_values, f)

            model.save(join(self.args.output_root, "models", f"testing_{new_model_name}.pth.tar"))

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


