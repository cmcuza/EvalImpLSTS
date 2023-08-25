import numpy as np
from typing import Tuple
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# ETTm1 {'n_estimators': 100, 'max_depth': 8, 'subsample': 1}


class XGBoost:
    def __init__(self, in_length, out_length, n_estimators, max_depth, subsample):
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            # 'min_child_weight': min_child_weight
        }
        self.input_seq_len = in_length
        self.target_sequence_length = out_length
        self.step_size = 1
        self.model = xgb.XGBRegressor(
                        n_estimators=self.hyperparameters["n_estimators"],
                        max_depth=self.hyperparameters["max_depth"],
                        subsample=self.hyperparameters["subsample"],
                        # min_child_weight=self.hyperparameters["min_child_weight"],
                        objective="reg:squarederror",
                        tree_method="hist")

    def get_xgboost_x_y(self, indices: list, data: np.array) -> Tuple[np.array, np.array]:
        print("Preparing data..")

        all_y = np.empty((len(indices), self.target_sequence_length))
        all_x = np.empty((len(indices), self.input_seq_len))
        # Loop over list of training indices
        data = np.squeeze(data)
        for i, idx in enumerate(indices):

            # Slice data into instance of length input length + target length
            data_instance = data[idx[0]:idx[1]]

            x = data_instance[0:self.input_seq_len]

            assert len(x) == self.input_seq_len

            y = data_instance[self.input_seq_len:self.input_seq_len + self.target_sequence_length]

            # Create all_y and all_x objects in first loop iteration
            all_y[i] = y.reshape(1, -1)
            all_x[i] = x.reshape(1, -1)

        print("Finished preparing data!")

        return all_x, all_y

    def get_indices_entire_sequence(self, data: pd.DataFrame) -> list:

        stop_position = len(data) - 1  # 1- because of 0 indexing

        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0

        subseq_last_idx = self.input_seq_len+self.target_sequence_length

        indices = []

        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx))

            subseq_first_idx += self.step_size

            subseq_last_idx += self.step_size

        return indices

    def train(self, train_data):
        training_indices = self.get_indices_entire_sequence(data=train_data)
        x_train, y_train = self.get_xgboost_x_y(indices=training_indices, data=train_data)

        self.model = MultiOutputRegressor(self.model).fit(x_train, y_train)

    def predict(self, test_data):
        test_indices = self.get_indices_entire_sequence(data=test_data)
        x_test, y_test = self.get_xgboost_x_y(indices=test_indices, data=test_data)
        return y_test, self.model.predict(x_test)
