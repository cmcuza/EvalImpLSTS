from darts.models import RNNModel
import numpy as np


class GRU:
    def __init__(self, input_len,
                 output_len,
                 parameters,
                 n_epochs,
                 optimizer_kwargs,
                 random_state,
                 torch_device_str,
                 lr_scheduler_cls,
                 lr_scheduler_kwargs,
                 model_name):

        self.input_len = input_len
        self.output_len = output_len

        self.model = RNNModel(
            input_chunk_length=input_len,
            output_chunk_length=output_len,
            model='GRU',
            d_model=parameters['d_model'],
            nhead=parameters['nhead'],
            n_rnn_layers=parameters['n_rnn_layers'],
            dropout=parameters['dropout'],
            hidden_dim=parameters['hidden_dim'],
            n_epochs=n_epochs,
            batch_size=32,
            nr_epochs_val_period=1,
            optimizer_kwargs=optimizer_kwargs,
            random_state=random_state,
            torch_device_str=torch_device_str,
            force_reset=True,
            log_tensorboard=True,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_name=model_name
        )

    def train(self, x_train, x_val):
        self.model.fit(x_train, val_series=x_val, verbose=True)

    def predict(self, x_test):
        backtest_test_en = self.model.historical_forecasts(series=x_test,
                                                           start=self.input_len,
                                                           forecast_horizon=self.output_len,
                                                           retrain=False,
                                                           verbose=False,
                                                           num_samples=1,
                                                           last_points_only=False)

        raw_p_values = np.asarray([e.all_values() for e in backtest_test_en]).squeeze()
        return raw_p_values[:-1, ...]

    def save(self, _dir):
        self.model.save_model(_dir)

