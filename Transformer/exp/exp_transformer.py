from exp_basic import ExpBasic
from Transformer.components.model import Transformer
from torch.optim.lr_scheduler import StepLR


class ExpMain(ExpBasic):
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
                            self.seq_len,
                            self.parameters,
                            self.args.train_epochs,
                            optimizer_kwargs,
                            self.random_state,
                            torch_device_str,
                            lr_scheduler_cls,
                            lr_scheduler_kwargs,
                            self.model_name)

        return model
