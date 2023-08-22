from os.path import join
from forecasting.GRU.exp.exp_gru import ExpGRU


def main_gru(args):
    exp = ExpGRU(args)
    exp.find_hyperparameters(join(args.root_path, 'pmc', args.data), 'gru')
    for itr in range(args.itr):
        exp.args.eblc = 'sz'
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_gru_exp_{itr}_rnd')
        exp.args.eblc = 'pmc'
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_gru_exp_{itr}_rnd')
        exp.args.eblc = 'swing'
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_gru_exp_{itr}_rnd')
        exp.model = None

