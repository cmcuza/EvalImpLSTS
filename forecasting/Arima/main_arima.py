from forecasting.Arima.exp.exp_arima import ExpArima
from forecasting.Arima.exp.exp_arima_mv import ExpArimaMV
from os.path import join


def main_arima(args):
    exp = ExpArima(args) if args.dataset != 'solar' else ExpArimaMV(args)
    print('Computing the predictions for SZ')
    exp.args.eblc = 'sz'
    data = join(args.root_path, 'sz', args.data)
    exp.run_exp(data, f'{args.dataset}_sz_raw_train_arima')
    print('Computing the predictions for PMC')
    exp.args.eblc = 'pmc'
    data = join(args.root_path, 'pmc', args.data)
    exp.run_exp(data, f'{args.dataset}_pmc_raw_train_arima')
    print('Computing the predictions for SWING')
    exp.args.eblc = 'swing'
    data = join(args.root_path, 'swing', args.data)
    exp.run_exp(data, f'{args.dataset}_swing_raw_train_arima')


