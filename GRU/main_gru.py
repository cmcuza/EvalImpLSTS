from os.path import join
from GRU.exp.exp_gru import ExpMain


def main_gru(args):
    exp = ExpMain(args)
    for itr in range(10):
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_gru_exp_{itr}_rnd')
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_gru_exp_{itr}_rnd')
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_gru_exp_{itr}_rnd')

