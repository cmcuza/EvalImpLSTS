import os
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

scaler = None


class CompressedDataset(Dataset):
    def __init__(self, root_path, flag, features, data_path, size, target, freq, eb, retrain):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if eb >= 1:
            eb = int(eb)

        self.unit = None if root_path.find('sz') != -1 else 'ms'
        self.criterio = (eb == 0 or self.set_type in [0, 1])
        if retrain == 1:
            self.criterio = (eb == 0)

        self.features = features
        self.target = target + ('-R' if self.criterio else f'-E{eb}')
        self.freq = freq
        self.eb = eb

        self.root_path = root_path
        self.data = data_path
        self.__read_data__()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def filter_df_raw(self, df_raw, columns):
        if self.criterio:
            eb_error_columns = list(filter(lambda c: c[-1] == 'R', columns))
        else:
            eb_error_columns = list(filter(lambda c: re.findall(f'E{self.eb}(?![0-9])', c), columns))
            if len(eb_error_columns) == 0:
                eb_error_columns = list(filter(lambda c: re.findall(f'E{self.eb*0.01}(?![0-9])', c), columns))
                if len(eb_error_columns) == 0:
                    eb_error_columns = list(filter(lambda c: re.findall(f'E{float(self.eb)}(?![0-9])', c), columns))
                    if len(eb_error_columns) == 0:
                        raise Exception('There is something wrong with the variables in the dataset')

        eb_error_columns = ['datetime'] + eb_error_columns

        df_raw = df_raw[eb_error_columns]

        return df_raw

    def scale(self, df_data, border1s, border2s):
        global scaler
        df_data = df_data.set_index('datetime')
        if scaler is None:
            scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = scaler.transform(df_data.values)

        return data


class ETT(CompressedDataset):
    def __init__(self, root_path,
                 flag='train',
                 size=(96, 48, 24),
                 features='S',
                 data='ettm1.parquet',
                 target='OT',
                 freq='t',
                 eb=0, retrain=0):

        super().__init__(root_path, flag, features, data, size, target, freq, eb, retrain)

    def __read_data__(self):
        global scaler

        print(os.path.join(self.root_path, self.data))
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data))
        # just for testing, delete

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], unit=self.unit)

        print(df_raw.head())

        columns = df_raw.columns

        df_raw = self.filter_df_raw(df_raw, columns)

        print('LOG: Data type', self.set_type, 'with columns', df_raw.columns, 'with target', self.target)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        # border1s = [0, 7000 - self.seq_len, 8000 - self.seq_len]
        # border2s = [7000, 8000, 10000]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = None

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.set_type == 0:
            scaler = None

        data = self.scale(df_data, border1s, border2s)

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime)
        data_stamp = time_features(df_stamp)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


class Weather(CompressedDataset):
    def __init__(self, root_path,
                 flag='train',
                 size=(96, 48, 24),
                 features='S',
                 data='weather.parquet',
                 target='OT',
                 freq='t',
                 eb=0,
                 retrain=0):

        super().__init__(root_path, flag, features, data, size, target, freq, eb, retrain)

    def __read_data__(self):
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data))

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], unit=self.unit)

        columns = df_raw.columns

        df_raw = self.filter_df_raw(df_raw, columns)

        print('LOG: Data type', self.set_type, 'with columns', df_raw.columns, 'with target', self.target)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(border1s)
        print(border2s)

        df_data = None
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = self.scale(df_data, border1s, border2s)

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime)
        data_stamp = time_features(df_stamp)
        print('Data stamp shape', data_stamp.shape)
        self.data_x = data[border1:border2]

        print('Data size', self.data_x.shape)
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp


class Solar(CompressedDataset):
    def __init__(self, root_path,
                 flag='train',
                 size=(96, 48, 24),
                 features='M',
                 data='solar.parquet',
                 target='',
                 freq='t',
                 eb=0,
                 retrain=0):

        super().__init__(root_path, flag, features, data, size, target, freq, eb, retrain)

    def __read_data__(self):
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data))

        if 'sz' in self.root_path:
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        else:
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], unit='ms')

        # df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        print(df_raw.head())

        columns = df_raw.columns

        df_raw = self.filter_df_raw(df_raw, columns)

        print('LOG: Data type', self.set_type, 'with columns', df_raw.columns, 'with target', self.target)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[:-1]
        df_data = df_raw[cols_data]

        data = self.scale(df_data, border1s, border2s)

        df_stamp = df_raw[['datetime']][border1:border2]
        # df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime)
        data_stamp = time_features(df_stamp)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_timestamp = df_stamp.set_index('datetime')


class Wind(CompressedDataset):
    def __init__(self, root_path,
                 flag='train',
                 size=(96, 48, 24),
                 features='MS',
                 data='wind.parquet',
                 target='active_power',
                 freq='t',
                 eb=0,
                 retrain=0):
        super().__init__(root_path, flag, features, data, size, target, freq, eb, retrain)

    def __read_data__(self):
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data))

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], unit=self.unit)

        columns = df_raw.columns

        df_raw = self.filter_df_raw(df_raw, columns)

        print('LOG: Data type', self.set_type, 'with columns', df_raw.columns, 'with target', self.target)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        print(border1s)
        print(border2s)

        df_data = None

        if self.features == 'MS':
            cols_data = df_raw.columns[:-1]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        self.target_index = np.where(df_data.columns.values == self.target)[0][0]
        data = self.scale(df_data, border1s, border2s)

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime, unit='ms')
        data_stamp = time_features(df_stamp)

        self.data_x = data[border1:border2]

        print('Data size', self.data_x.shape)

        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp


class AUSElecDem(CompressedDataset):
    def __init__(self, root_path,
                 flag='train',
                 size=(96, 48, 24),
                 features='S',
                 data='aus_electrical_demand.parquet',
                 target='y',
                 freq='30T',
                 eb=0,
                 retrain=0):
        eb = float(eb)
        super().__init__(root_path, flag, features, data, size, target, freq, eb, retrain)

    def __read_data__(self):
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data))

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], unit=self.unit)

        # columns = df_raw.columns

        # df_raw = self.filter_df_raw(df_raw, columns)

        print('LOG: Data type', self.set_type, 'with columns', df_raw.columns, 'with target', self.target)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = int(len(df_raw) - num_train - num_test)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(border1s)
        print(border2s)

        df_data = None

        if self.features == 'MS':
            cols_data = df_raw.columns[:-1]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            try:
                df_data = df_raw[[self.target]]
            except KeyError:
                self.target = self.target.split('-')[0] + ('-R' if self.criterio else f'-E{float(self.eb)}')
                df_data = df_raw[[self.target]]

        self.target_index = np.where(df_data.columns.values == self.target)[0][0]
        data = self.scale(df_data, border1s, border2s)

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime, unit='ms')
        data_stamp = time_features(df_stamp)

        self.data_x = data[border1:border2]

        print('Data size', self.data_x.shape)

        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

