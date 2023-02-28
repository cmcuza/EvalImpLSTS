from NBeats.components.model import NBeats
from exp_basic import ExpBasic
from torch.optim.lr_scheduler import StepLR
from utils.metrics import *


class ExpMain(ExpBasic):
    def __init__(self, args):

        super().__init__(args)
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

        self.model_name = "_".join([self.model_name,
                                   str(self.random_state),
                                   self.args.data.strip('.parquet'),
                                   str(self.seq_len),
                                   str(self.pred_len)] +
                                  [str(e) for e in list(self.parameters.values())])

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
