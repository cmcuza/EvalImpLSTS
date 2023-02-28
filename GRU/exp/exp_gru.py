from exp_basic import ExpBasic
from GRU.components.model import GRU
from torch.optim.lr_scheduler import StepLR


class ExpMain(ExpBasic):
    def __init__(self, args):

        super().__init__(args)
        self.parameters['hidden_dim'] = args.hidden_dim
        self.parameters['n_rnn_layers'] = args.n_rnn_layers
        self.parameters['dropout'] = args.dropout
        self.parameters['lr'] = args.learning_rate
        self.parameters['weight_decay'] = args.weight_decay

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

        model = GRU(self.seq_len,
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
