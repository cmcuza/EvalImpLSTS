from forecasting.NBeats.components.nbeats_model import NBeats
from exp_basic import ExpBasic
from torch.optim.lr_scheduler import StepLR
from itertools import product


class ExpNBeats(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        # {'num_stacks': 30, 'num_blocks': 2, 'num_layers': 4, 'layer_widths': 32, 'dropout': 0}
        self.parameters['weight_decay'] = args.weight_decay
        self.parameters['num_stacks'] = args.num_stacks
        self.parameters['num_blocks'] = args.num_blocks
        self.parameters['num_layers'] = args.num_layers
        self.parameters['layer_widths'] = args.layer_widths
        self.parameters['dropout'] = args.dropout
        self.parameters['lr'] = args.learning_rate

    def _build_model(self):
        lr_scheduler_cls = StepLR
        lr_scheduler_kwargs = {'step_size': 2, 'gamma': 0.5}

        self.model_name = self.build_name(self.model_name)

        torch_device_str = 'cuda' if self.args.use_gpu else 'cpu'
        optimizer_kwargs = {"lr": self.parameters['lr'], 'weight_decay': self.parameters['weight_decay']}

        model = NBeats(self.seq_len,
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

    def build_name(self, cname):
        return "_".join([cname,
                         str(self.random_state),
                         self.args.data.strip('.parquet'),
                         str(self.seq_len),
                         str(self.pred_len)] +
                         [str(e) for e in list(self.parameters.values())])

    def change_hyperparameters(self):
        num_stacks = [15, 30]
        num_blocks = [1, 2]
        num_layers = [2, 4]
        layer_widths = [32, 64]
        dropout = [0, 0.05]

        for i, (ns, nb, nl, lw, dp) in enumerate(product(num_stacks, num_blocks, num_layers, layer_widths, dropout)):
            self.parameters['num_stacks'] = ns
            self.parameters['num_blocks'] = nb
            self.parameters['num_layers'] = nl
            self.parameters['layer_widths'] = lw
            self.parameters['dropout'] = dp
            yield i

    def set_best_hyperparameter(self, index):
        num_stacks = [15, 30]
        num_blocks = [1, 2]
        num_layers = [2, 4]
        layer_widths = [32, 64]
        dropout = [0, 0.05]

        ns, nb, nl, lw, dp = list(product(num_stacks, num_blocks, num_layers, layer_widths, dropout))[index]
        self.parameters['num_stacks'] = ns
        self.parameters['num_blocks'] = nb
        self.parameters['num_layers'] = nl
        self.parameters['layer_widths'] = lw
        self.parameters['dropout'] = dp
        print('Best hyperparameters are', self.parameters)
