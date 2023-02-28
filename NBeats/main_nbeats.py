from os.path import join
from NBeats.exp.exp_nbeats import ExpMain


def main_nbeats(args):
    exp = ExpMain(args)
    for itr in range(10):
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_nbeats_exp_{itr}_rnd')
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_nbeats_exp_{itr}_rnd')
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_nbeats_exp_{itr}_rnd')

