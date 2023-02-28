from os.path import join
from Transformer.exp.exp_transformer import ExpMain


def main_transformer(args):
    str_train = 'raw' if args.train_raw else 'dec'
    exp = ExpMain(args)
    for itr in range(10):
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.data}_sz_{str_train}_train_transformer_exp_{itr}_rnd_')
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.data}_pmc_{str_train}_train_transformer_exp_{itr}_rnd_')
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.data}_swing_{str_train}_train_transformer_exp_{itr}_rnd_')

