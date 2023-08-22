from os.path import join
from forecasting.NBeats.exp.exp_nbeats import ExpNBeats


def main_nbeats(args):
    exp = ExpNBeats(args)
    exp.find_hyperparameters(join(args.root_path, 'pmc', args.data), 'nbeats')
    for itr in range(args.itr):
        exp.args.eblc = 'sz'
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_nbeats_exp_{itr}_rnd')
        exp.args.eblc = 'pmc'
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_nbeats_exp_{itr}_rnd')
        exp.args.eblc = 'swing'
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_nbeats_exp_{itr}_rnd')
        exp.model = None