from forecasting.ARIMA.exp.ExpMain import ExpMain
from os.path import join


def main_arima(args):
    exp = ExpMain(args)
    data = join(args.root_path, 'sz', args.data)
    exp.run_exp(data, f'{args.dataset}_sz_raw_train_arima')
    data = join(args.root_path, 'pmc', args.data)
    exp.run_exp(data, f'{args.dataset}_pmc_raw_train_arima')
    data = join(args.root_path, 'swing', args.data)
    exp.run_exp(data, f'{args.dataset}_swing_raw_train_arima')


