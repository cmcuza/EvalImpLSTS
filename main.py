from forecasting.DLinear.main_dlinear import main_dlinear
from forecasting.GRU.main_gru import main_gru
from forecasting.NBeats.main_nbeats import main_nbeats
from forecasting.Transformer.main_transformer import main_transformer
from forecasting.Arima.main_arima import main_arima
from forecasting.Informer.main_informer import main_informer
from forecasting.XGBoost.main_xgboost import main_xgboost
from os.path import join
import argparse
import torch
import random
import numpy as np


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
parser.add_argument('--model_id', type=str, required=True, default='gru', help='model id')
parser.add_argument('--retrain', type=int, required=False, default=0, help='model id')

# data loader
parser.add_argument('--data', type=str, required=True, default='ettm1', help='dataset type')
parser.add_argument('--root_path', type=str, default=join('.', 'data', 'compressed'), help='root path of the data file')
parser.add_argument('--eblc', type=str, default='pmc', help='error bound lossy compressor')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; '
                                                              'M:multivariate predict multivariate, '
                                                              'S:univariate predict univariate, '
                                                              'MS:multivariate predict univariate')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:'
                                                          '[s:secondly, t:minutely, h:hourly, '
                                                          'd:daily, b:business days, '
                                                          'w:weekly, m:monthly], '
                                                          'you can also use more '
                                                          'detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default=join('.', 'checkpoints'), help='location of model checkpoints')
parser.add_argument('--output_root', type=str, default=join('.', 'output'), help='location of model checkpoints')
parser.add_argument('--eb', type=float, default=0.0)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--target_var', type=str, required=True, default='OT', help='target variable in the dataset')
parser.add_argument('--train_raw', type=bool, default=True, help='training on raw data')
parser.add_argument('--EB', nargs='+', type=float, default=[0, 1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80],
                    help='error bounds to run the experiments on')


parser.add_argument('--individual', action='store_true', default=True, help='DLinear: a linear layer for '
                                                                            'each variate(channel) individually')

# DLinear & Informer & Transformer
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + '
                                                              'temporal embedding + '
                                                              'positional embedding 2: value embedding + '
                                                              'temporal embedding 3: value embedding + '
                                                              'positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # Change to Solar
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)


parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# GRU Model
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden layers dimension')
parser.add_argument('--n_rnn_layers', type=int, default=2, help='num of heads')

# NBeats Model
parser.add_argument('--num_stacks', type=int, default=15, help='hidden layers dimension')
parser.add_argument('--num_blocks', type=int, default=1, help='num of heads')
parser.add_argument('--num_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--layer_widths', type=int, default=64, help='num of decoder layers')

# XGBoost

parser.add_argument('--n_estimators', type=int, default=100, help='number of estimators')
parser.add_argument('--max_depth', type=int, default=6, help='Maximum tree depth for base learners')
parser.add_argument('--subsample', type=float, default=0.8, help='subsample')
parser.add_argument('--min_child_weight', type=float, default=1, help='Minimum sum of instance weight(hessian) '
                                                                     'needed in a child')


# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='optimizer weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--few', type=bool, default=True, help='Consider only 2 or 4 temporal features')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

args.dataset = args.data.split('_')[0]

if __name__ == '__main__':
    if 'arima' in args.model_id:
        main_arima(args)
    elif 'dlinear' in args.model_id:
        main_dlinear(args)
    elif 'xgboost' in args.model_id:
        main_xgboost(args)
    elif 'nbeats' in args.model_id:
        main_nbeats(args)
    elif 'transformer' in args.model_id:
        main_transformer(args)
    elif 'informer' in args.model_id:
        main_informer(args)
    elif 'gru' in args.model_id:
        main_gru(args)
    else:
        print('----Model not supported----')
        print('Supported model id: [dlinear, nbeats, transformer, informer, gru]')

