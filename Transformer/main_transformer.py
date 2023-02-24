import pandas as pd
import pickle as pkl
from darts import TimeSeries
from Transformer.components.model import Transformer
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from utils.metrics import *


def temporal_train_val_test_split(data, seq_len, ot='OT-R'):
    border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    train_border1 = border1s[0]
    train_border2 = border2s[0]
    target_column = ['datetime', ot]

    train_data = data[target_column][train_border1:train_border2].reset_index(drop=True)

    val_border1 = border1s[1]
    val_border2 = border2s[1]

    val_data = data[target_column][val_border1:val_border2].reset_index(drop=True)

    test_border1 = border1s[2]
    test_border2 = border2s[2]

    test_data = data[target_column][test_border1:test_border2].reset_index(drop=True)

    return train_data, val_data, test_data


def create_sequence(data, sequence_window, forecasting_horizon):
    x_seq_data = []
    y_seq_data = []
    for i in range(data.shape[0] - sequence_window - forecasting_horizon):
        train_seq = data.iloc[i:i + sequence_window]
        train_label = data.iloc[i + sequence_window:i + sequence_window + forecasting_horizon]
        x_seq_data.append(TimeSeries.from_dataframe(train_seq, time_col='datetime'))
        y_seq_data.append(TimeSeries.from_dataframe(train_label, time_col='datetime'))

    return x_seq_data, y_seq_data


def run_baseline_experiment(data, input_len, model_name, output_len, parameters):
    print("Running testing", model_name, "on", data, "with", input_len, "and", output_len, 'with', parameters)
    print("Loading the data")
    full_dataset = pd.read_parquet(data)
    full_dataset['datetime'] = pd.to_datetime(full_dataset['datetime'])

    columns = full_dataset.columns
    raw_columns = list(filter(lambda c: c[-1] == 'R', columns))
    raw_columns = ['datetime'] + raw_columns
    temp_full_dataset = full_dataset[raw_columns].copy()
    all_train_data, all_val_data, all_test_data = temporal_train_val_test_split(temp_full_dataset, input_len)

    print('train size', all_train_data[0].shape)
    print(all_train_data[0].head())
    x_train = TimeSeries.from_dataframe(all_train_data[0], time_col='datetime')

    print('val size', all_val_data[0].shape)
    print(all_val_data[0].head())
    x_val = TimeSeries.from_dataframe(all_val_data[0], time_col='datetime')

    scaler = Scaler(StandardScaler())
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    lr_scheduler_cls = StepLR
    lr_scheduler_kwargs = {'step_size': 2, 'gamma': 0.5}

    random_state = np.random.randint(1000)

    model_name = "_".join([model_name + str(random_state),
                           data.split('/')[-1],
                           str(input_len),
                           str(output_len)] +
                          [str(e) for e in list(parameters.values())] +
                          ['eb_0.0'])

    num_samples = 1

    torch_device_str = 'cpu'
    n_epochs = 10
    optimizer_kwargs = {"lr": 1e-4, 'weight_decay': parameters['weight_decay']}

    target_model = 'transformer'
    model = Transformer(input_len,
                        output_len,
                        parameters,
                        n_epochs,
                        optimizer_kwargs,
                        random_state,
                        torch_device_str,
                        lr_scheduler_cls,
                        lr_scheduler_kwargs,
                        model_name)

    print("Training")
    model.train(x_train, x_val)

    print("Testing")

    for eb in [0, 1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80]:
        if eb != 0:
            if data.find('sz') != -1:
                eb = eb * 0.01
                eb_error_columns = list(filter(lambda c: c.split('-')[-1] == f'E{eb}', columns))
            else:
                eb_error_columns = list(filter(lambda c: c.split('-')[-1] == f'E{eb}', columns))

            eb_error_columns.append('datetime')
            temp_full_dataset = full_dataset[eb_error_columns].copy()
            _, _, all_test_data = temporal_train_val_test_split(temp_full_dataset, input_len, ot=f'OT-E{eb}')
            model_name = "_".join([model_name + str(random_state),
                                   'ettm',
                                   str(input_len),
                                   str(output_len)] +
                                  [str(e) for e in list(parameters.values())] +
                                  ['eb', str(eb)])

        print('test size', all_test_data[0].shape)
        print(all_test_data[0].head())
        x_test = TimeSeries.from_dataframe(all_test_data[0], time_col='datetime')
        x_test = scaler.transform(x_test)

        raw_p_values = model.predict(x_test)

        seq_x_test, seq_y_test = create_sequence(all_test_data[0], input_len, output_len)

        print("Computing metrics")

        raw_y_values = np.asarray([scaler.transform(e).all_values() for e in seq_y_test]).squeeze()

        results = {
            "MAE": np.round(MAE(raw_y_values, raw_p_values), 4),
            "MSE": np.round(MSE(raw_y_values, raw_p_values), 4),
            "RMSE": np.round(RMSE(raw_y_values, raw_p_values), 4)
        }

        print("Results", results)

        results_root = f"../output/{target_model}/"

        type_results = 'train_raw_test_dec/'

        pd.DataFrame(results, index=[0]).to_csv(results_root + "results/" + type_results + f"testing_{model_name}.csv", index=None)

        with open(results_root + f"predictions/" + type_results + f"/testing_{model_name}_output.pickle", 'wb') as f:
            pkl.dump(raw_p_values, f)

        with open(results_root + f"predictions/" + type_results + f"testing_{model_name}_true.pickle", 'wb') as f:
            pkl.dump(raw_y_values, f)

        model.save(results_root + "models/" + type_results + f"testing_{model_name}.pth.tar")


if __name__ == "__main__":

    parameters = {}
    input_len = 96
    output_len = 24
    eb = 0.0

    parameters['weight_decay'] = 0.0001
    train_raw = True
    str_train = 'raw' if train_raw else 'dec'
    np.random.seed(42)

    d_model = 32
    dropout = 0.0
    n_head = 8
    num_enc_dec_layers = 2
    dim_feedforward = 64
    for exp in range(10):
        parameters['d_model'] = d_model
        parameters['num_encoder_layers'] = num_enc_dec_layers
        parameters['num_decoder_layers'] = num_enc_dec_layers
        parameters['dim_feedforward'] = dim_feedforward
        parameters['dropout'] = dropout
        parameters['nhead'] = n_head
        data = '../data/compressed/sz/ETTm1.parquet'
        run_baseline_experiment(data,
                                input_len,
                                f'ettm1_sz_{str_train}_train_transformer_exp_{exp}_rnd_',
                                output_len, parameters)
        data = '../data/compressed/pmc/ETTm1.parquet'
        run_baseline_experiment(data,
                                input_len,
                                f'ettm1_pmc_{str_train}_train_transformer_exp_{exp}_rnd_',
                                output_len,
                                parameters)
        data = '../data/compressed/swing/ETTm1.parquet'
        run_baseline_experiment(data,
                                input_len,
                                f'ettm1_swing_{str_train}_train_transformer_exp_{exp}_rnd_',
                                output_len,
                                parameters)

