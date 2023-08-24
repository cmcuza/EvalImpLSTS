from os.path import join
from forecasting.GRU.exp.exp_gru import ExpGRU


def main_gru(args):
    exp = ExpGRU(args)
    # exp.find_hyperparameters(join(args.root_path, 'pmc', args.data), 'gru')
    # Best hyperparameters are {'hidden_dim': 64, 'n_rnn_layers': 3, 'dropout': 0.05} for aus_electrical_demand
    for itr in range(args.itr):
        print('Computing the predictions for SZ')
        exp.args.eblc = 'sz'
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_gru_exp_{itr}_rnd')
        print('Computing the predictions for PMC')
        exp.args.eblc = 'pmc'
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_gru_exp_{itr}_rnd')
        print('Computing the predictions for SWING')
        exp.args.eblc = 'swing'
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_gru_exp_{itr}_rnd')
        exp.model = None

