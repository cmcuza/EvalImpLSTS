from os.path import join
from forecasting.Transformer.exp.exp_transformer import ExpTransformer


def main_transformer(args):
    # str_train = 'raw' if args.train_raw else 'dec'
    exp = ExpTransformer(args)
    exp.find_hyperparameters(join(args.root_path, 'pmc', args.data), 'transformer')
    for itr in range(args.itr):
        exp.args.eblc = 'sz'
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.data}_sz_raw_train_transformer_exp_{itr}_rnd_')
        exp.args.eblc = 'pmc'
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.data}_pmc_raw_train_transformer_exp_{itr}_rnd_')
        exp.args.eblc = 'swing'
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.data}_swing_raw_train_transformer_exp_{itr}_rnd_')

