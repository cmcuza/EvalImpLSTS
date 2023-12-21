from os.path import join
from forecasting.XGBoost.exp.exp_xgboost import ExpXGBoost
from forecasting.XGBoost.exp.exp_xgboost_mv import ExpXGBoostMV


def main_xgboost(args):
    exp = ExpXGBoost(args) if args.dataset != 'solar' else ExpXGBoostMV(args)
    print('Computing the predictions for SZ')
    exp.args.eblc = 'sz'
    data = join(args.root_path, 'pmc', args.data)
    exp.find_hyperparameters(data, 'xgboost')
    exp.run_exp(data, f'{args.dataset}_sz_raw_train_xgboost')
    print('Computing the predictions for PMC')
    exp.args.eblc = 'pmc'
    data = join(args.root_path, 'pmc', args.data)
    exp.run_exp(data, f'{args.dataset}_pmc_raw_train_xgboost')
    print('Computing the predictions for SWING')
    exp.args.eblc = 'swing'
    data = join(args.root_path, 'swing', args.data)
    exp.run_exp(data, f'{args.dataset}_swing_raw_train_xgboost')

