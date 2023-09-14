from os.path import join
from forecasting.Transformer.exp.exp_transformer import ExpTransformer


def main_transformer(args):
    exp = ExpTransformer(args)
    exp.find_hyperparameters(join(args.root_path, 'pmc', args.data), 'transformer')
    # Best hyperparameters are {'d_model': 64, 'num_encoder_layers': 2,
    # 'num_decoder_layers': 2, 'dim_feedforward': 512, 'dropout': 0.05, 'nhead': 8} for aus
    for itr in range(args.itr):
        print('Computing the predictions for SZ')
        exp.args.eblc = 'sz'
        data = join(args.root_path, 'sz', args.data)
        exp.run_exp(data, f'{args.dataset}_sz_raw_train_transformer_exp_{itr}_rnd_')
        print('Computing the predictions for PMC')
        exp.args.eblc = 'pmc'
        data = join(args.root_path, 'pmc', args.data)
        exp.run_exp(data, f'{args.dataset}_pmc_raw_train_transformer_exp_{itr}_rnd_')
        print('Computing the predictions for SWING')
        exp.args.eblc = 'swing'
        data = join(args.root_path, 'swing', args.data)
        exp.run_exp(data, f'{args.dataset}_swing_raw_train_transformer_exp_{itr}_rnd_')
        exp.model = None

