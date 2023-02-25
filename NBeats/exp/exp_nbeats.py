from os.path import join, isdir
import os
from darts import TimeSeries
import pandas as pd
import pickle as pkl
from data.data_factory import data_provider
from NBeats.components.model import NBeats
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from utils.metrics import *


class ExpMain:
    def __init__(self, args):
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.ot = args.target_var
        self.parameters = dict()
        self.parameters['weight_decay'] = args.weight_decay
        self.parameters['d_model'] = args.d_model
        self.parameters['num_encoder_layers'] = args.e_layers
        self.parameters['num_decoder_layers'] = args.d_layers
        self.parameters['dim_feedforward'] = args.d_ff
        self.parameters['dropout'] = args.dropout
        self.parameters['nhead'] = args.n_heads
        self.parameters['lr'] = args.lr
        self.args = args
        if self.args.data.find('ETT') != -1:
            self.fixed_borders = True
        else:
            self.fixed_borders = False

        if not isdir(self.args.output_root):
            os.makedirs(self.args.output_root)
            os.makedirs(join(self.args.output_root, 'predictions'))
            os.makedirs(join(self.args.output_root, 'models'))

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def create_sequence(self, data):
        x_seq_data = []
        y_seq_data = []
        for i in range(data.shape[0] - self.input_len - self.output_len):
            train_seq = data.iloc[i:i + self.input_len]
            train_label = data.iloc[i + self.input_len:i + self.input_len + self.output_len]
            x_seq_data.append(TimeSeries.from_dataframe(train_seq, time_col='datetime'))
            y_seq_data.append(TimeSeries.from_dataframe(train_label, time_col='datetime'))

        return x_seq_data, y_seq_data

    def temporal_train_val_test_split(self, data, eb):

        if self.fixed_borders:
            # border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
            # border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1s = [0, 6000-self.input_len, 8000-self.input_len]
            border2s = [6000, 8000, 10000]
        else:
            num_train = int(len(data) * 0.7)
            num_test = int(len(data) * 0.2)
            num_val = len(data) - num_train - num_test
            border1s = [0, num_train - self.input_len, len(data) - num_test - self.input_len]
            border2s = [num_train, num_train + num_val, len(data)]

        train_border1 = border1s[0]
        train_border2 = border2s[0]
        target_column = ['datetime', self.ot+'-'+eb]

        train_data = data[target_column][train_border1:train_border2].reset_index(drop=True)

        val_border1 = border1s[1]
        val_border2 = border2s[1]

        val_data = data[target_column][val_border1:val_border2].reset_index(drop=True)

        test_border1 = border1s[2]
        test_border2 = border2s[2]

        test_data = data[target_column][test_border1:test_border2].reset_index(drop=True)

        return train_data, val_data, test_data

    def run_exp(self, data, model_name):
        print("Running testing", model_name, "on", data, "with", self.input_len, "and", self.output_len)
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

        model = self._build_model(model_name)

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

            new_model_name = "_".join([model_name, str(self.random_state),
                                   self.args.data,
                                   str(self.input_len),
                                   str(self.output_len)] +
                                  [str(e) for e in list(self.parameters.values())] +
                                  ['eb', str(eb)])

            pd.DataFrame(results, index=[0]).to_csv(join(self.args.output_root, f"testing_{new_model_name}.csv"), index=False)

            with open(join(self.args.output_root, "predictions", f"testing_{new_model_name}_output.pickle"), 'wb') as f:
                pkl.dump(raw_p_values, f)

            model.save(join(self.args.output_root, "models", f"testing_{new_model_name}.pth.tar"))

    def _build_model(self, model_name):
        lr_scheduler_cls = StepLR
        lr_scheduler_kwargs = {'step_size': 2, 'gamma': 0.5}

        self.random_state = np.random.randint(1000)

        new_model_name = "_".join([model_name,
                                   str(self.random_state),
                                   self.args.data.strip('.parquet'),
                                   str(self.input_len),
                                   str(self.output_len)] +
                                  [str(e) for e in list(self.parameters.values())])

        torch_device_str = 'cuda' if self.args.use_gpu else 'cpu'
        optimizer_kwargs = {"lr": self.parameters['lr'], 'weight_decay': self.parameters['weight_decay']}

        model = NBeats(self.input_len,
                       self.output_len,
                       self.parameters,
                       self.args.train_epochs,
                       optimizer_kwargs,
                       self.random_state,
                       torch_device_str,
                       lr_scheduler_cls,
                       lr_scheduler_kwargs,
                       new_model_name)

        return model
