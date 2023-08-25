from exp_basic import ExpBasic
from forecasting.Transformer.components.transformer_model import Transformer
from torch.optim.lr_scheduler import StepLR
from itertools import product


class ExpTransformer(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.parameters['weight_decay'] = args.weight_decay
        self.parameters['d_model'] = args.d_model
        self.parameters['num_encoder_layers'] = args.e_layers
        self.parameters['num_decoder_layers'] = args.d_layers
        self.parameters['dim_feedforward'] = args.d_ff
        self.parameters['dropout'] = args.dropout
        self.parameters['nhead'] = args.n_heads
        self.parameters['lr'] = args.learning_rate

    def _build_model(self):
        lr_scheduler_cls = StepLR
        lr_scheduler_kwargs = {'step_size': 2, 'gamma': 0.5}

        self.model_name = "_".join([self.model_name,
                                    str(self.random_state),
                                    str(self.seq_len),
                                    str(self.pred_len)] +
                                   [str(e) for e in list(self.parameters.values())])

        torch_device_str = 'cuda' if self.args.use_gpu else 'cpu'
        optimizer_kwargs = {"lr": self.parameters['lr'], 'weight_decay': self.parameters['weight_decay']}

        model = Transformer(self.seq_len,
                            self.pred_len,
                            self.parameters,
                            self.args.train_epochs,
                            optimizer_kwargs,
                            self.random_state,
                            torch_device_str,
                            lr_scheduler_cls,
                            lr_scheduler_kwargs,
                            self.model_name)

        return model

    def change_hyperparameters(self):
        d_model = [32, 64]
        num_encoder_layers = [2, 3]
        dim_feedforward = [256, 512]
        dropout = [0, 0.05]
        nhead = [4, 8]

        for i, (dm, nel, df, nh, dp) in enumerate(product(d_model, num_encoder_layers, dim_feedforward, nhead, dropout)):
            self.parameters['d_model'] = dm
            self.parameters['num_encoder_layers'] = nel
            self.parameters['num_decoder_layers'] = nel
            self.parameters['dim_feedforward'] = df
            self.parameters['dropout'] = dp
            self.parameters['nhead'] = nh
            yield i

    def set_best_hyperparameter(self, index):
        d_model = [32, 64]
        num_encoder_layers = [2, 3]
        dim_feedforward = [256, 512]
        dropout = [0, 0.05]
        nhead = [4, 8]

        (dm, nel, df, nh, dp) = list(product(d_model, num_encoder_layers, dim_feedforward, nhead, dropout))[index]
        self.parameters['d_model'] = dm
        self.parameters['num_encoder_layers'] = nel
        self.parameters['num_decoder_layers'] = nel
        self.parameters['dim_feedforward'] = df
        self.parameters['dropout'] = dp
        self.parameters['nhead'] = nh

        print('Best hyperparameters are', self.parameters)
