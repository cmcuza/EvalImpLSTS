import argparse
from os.path import join
import torch
import random
import numpy as np
from NBeats.exp.exp_nbeats import ExpMain


fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='NBeats family for Time Series Forecasting')

# basic config
parser.add_argument('--train_raw', type=bool, default=True, help='train on raw time series')
parser.add_argument('--model', type=str, default='NBeats', help='model name, options: [NBeats]')
parser.add_argument('--output_root', type=str, default=join('..', 'output', 'nbeats'), help='results folder')
# data loader
parser.add_argument('--root_path', type=str, default=join('..', 'data', 'compressed'), help='root path of the data file')
parser.add_argument('--data', type=str, required=True, default='ETTm1.parquet', help='data file')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--eb', type=float, default=0.0)
# forecasting task
parser.add_argument('--input_len', type=int, default=96, help='input sequence length')
parser.add_argument('--output_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--target_var', type=str, required=True, default='OT', help='target variable in the dataset')
parser.add_argument('--EB', type=list, default=[0, 1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80], help='error bounds to run the experiments on')
# Model
parser.add_argument('--num_stacks', type=int, default=15, help='hidden layers dimension')
parser.add_argument('--num_blocks', type=int, default=1, help='num of heads')
parser.add_argument('--num_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--layer_widths', type=int, default=64, help='num of decoder layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

# optimization
parser.add_argument('--itr', type=int, default=10, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if __name__ == "__main__":
    str_train = 'raw' if args.train_raw else 'dec'
    np.random.seed(42)
    exp = ExpMain(args)
    for itr in range(10):
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.data}_sz_{str_train}_train_nbeats_exp_{itr}_rnd_')
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.data}_pmc_{str_train}_train_nbeats_exp_{itr}_rnd_')
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.data}_swing_{str_train}_train_nbeats_exp_{itr}_rnd_')

