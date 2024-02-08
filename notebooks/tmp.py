import pandas as pd
import numpy as np


def get_baselines_well(features):
    num_colums = features.columns[:-3]
    features[num_colums] = features[num_colums].astype(float)
    raw_feat = features[features.compression == 'RAW']
    raw_feat = raw_feat.append([raw_feat, raw_feat], ignore_index=True)
    features.drop(features[features.compression == 'RAW'].index)
    raw_feat.at[0, 'compression'] = 'PMC'
    raw_feat.at[1, 'compression'] = 'SZ'
    raw_feat.at[2, 'compression'] = 'SWING'
    return pd.concat([raw_feat, features])


def get_feature_diff(features):
    raw_feat = features.iloc[0]
    num_colums = features.columns[:-3]
    features[num_colums] = features[num_colums].astype(float)
    features[num_colums] -= raw_feat[num_colums].astype(float)
    features[num_colums] = features[num_colums].astype(float)
    raw_feat = features[features.compression == 'RAW']
    raw_feat = raw_feat.append([raw_feat], ignore_index=True)
    features.at[0, 'compression'] = 'PMC'
    raw_feat.at[0, 'compression'] = 'SZ'
    raw_feat.at[1, 'compression'] = 'SWING'
    return pd.concat([raw_feat, features])


def get_baselines_well_solar(features):
    num_colums = features.columns[2:-1]
    features[num_colums] = features[num_colums].astype(float)
    raw_feat = features[features.compression == 'RAW']
    raw_feat = raw_feat.append([raw_feat, raw_feat], ignore_index=True)
    features.drop(features[features.compression == 'RAW'].index)
    raw_feat.at[0, 'compression'] = 'PMC'
    raw_feat.at[1, 'compression'] = 'SZ'
    raw_feat.at[2, 'compression'] = 'SWING'
    return pd.concat([raw_feat, features])


def get_features_diff_solar(features):
    raw_feat = features.iloc[0]
    num_colums = features.columns[2:-1]
    features[num_colums] = features[num_colums].astype(float)
    features[num_colums] -= raw_feat[num_colums].astype(float)
    features[num_colums] = features[num_colums].astype(float)
    raw_feat = features[features.compression == 'RAW']
    raw_feat = raw_feat.append([raw_feat], ignore_index=True)
    features.at[0, 'compression'] = 'PMC'
    raw_feat.at[0, 'compression'] = 'SZ'
    raw_feat.at[1, 'compression'] = 'SWING'
    return pd.concat([raw_feat, features])


def get_tfe(model_results):
    baseline_result  = model_results[model_results.eblc == 'baseline']
    model_results['TFE'] = (model_results.nrmse.values - baseline_result.nrmse.values)/baseline_result.nrmse.values
    baseline_result = baseline_result.append([baseline_result]*2, ignore_index=True)
    baseline_result.at[0, 'eblc'] = 'pmc'
    baseline_result.at[1, 'eblc'] = 'sz'
    baseline_result.at[2, 'eblc'] = 'swing'
    baseline_result['TFE'] = 0.
    return pd.concat([model_results, baseline_result])

def join_with_arima():
    df = pd.read_csv('../results/tfe/per_model/arima_results.csv')
    df = df[['error', 'nrmse', 'eblc', 'data', 'eb']]
    arima_results = pd.DataFrame()
    for data in df.data.unique():
        data_results = df[df.data == data]
        get_tfe(data_results)
        arima_results = pd.concat([arima_results, data_results])
    arima_results.reset_index(drop=True, inplace=True)
    arima_results.eblc = arima_results.eblc.str.upper()
    arima_results['model'] = 'ARIMA'
    arima_results.data = arima_results.data.str.upper()
    arima_results['tfe_metric'] = 'nrmse'
    arima_results['te_metric'] = 'nrmse'
    arima_results['te'] = arima_results['error']
    x = arima_results.eb.values
    arima_results.eb = np.where(x >= 1.0, x*0.01, x)
    arima_results = arima_results.rename({'nrmse': 'forecasting error', 'eblc': 'compression'}, axis=1)
    all_model_results = pd.read_csv('../results/tfe/all_models_results.csv')
    cr_df = pd.read_csv('../results/cr/all_cr.csv')
    cr_df.compression = cr_df.compression.str.upper()
    cr_df.data = cr_df.data.str.upper()
    cr_df.rename({'gzip': 'compression ratio', 'error_bound': 'eb'}, axis=1, inplace=True)
    cr_df.set_index(['data', 'compression', 'eb'], inplace=True)
    joined = cr_df.merge(arima_results, on=['data', 'compression', 'eb'])
    namr = pd.concat([all_model_results, joined]).reset_index(drop=True)
    namr.to_csv('../results/tfe/all_models_results_arima.csv', index=False)


def compute_features_tfe_corr_xg():
    all_models_results = pd.read_csv('../results/tfe/models_results_xg.csv')
    datasets = all_models_results.data.str.lower().unique()
    all_datasets_features = pd.DataFrame()
    for dataset in datasets:
        print(dataset)
        features = pd.read_csv(f'../results/features/test_{dataset}_features.csv')
        x = features.eb.values
        features.eb = np.where(x >= 1.0, x*0.01, x)
        features.eblc = features.eblc.str.upper()
        features.rename({'eblc': 'compression'}, axis=1, inplace=True)
        features = get_feature_diff(features) if 'solar' not in dataset else get_features_diff_solar(features)
        features.set_index(['compression', 'eb'], inplace=True)
        model_results = all_models_results[all_models_results.data == dataset.upper()]
        model_results = model_results[model_results.te_metric == 'nrmse']
        model_results = model_results[['eb',  'te', 'TFE', 'compression', 'model']]
        joined_df = features.merge(model_results.reset_index(drop=True), on=['compression', 'eb']).drop_duplicates()
        joined_df['data'] = dataset
        all_datasets_features = pd.concat([all_datasets_features, joined_df])

    all_datasets_features.to_csv('../results/features/all_datasets_features_diff_xg.csv', index=False)
    return all_datasets_features.corr('spearman')


def compute_features_with_xgboost():
    all_models_results = pd.read_csv('../results/tfe/models_results_xg.csv')
    datasets = all_models_results.data.str.lower().unique()
    all_datasets_features = pd.DataFrame()
    for dataset in datasets:
        print(dataset)
        features = pd.read_csv(f'../results/features/test_{dataset}_features.csv')
        x = features.eb.values
        features.eb = np.where(x >= 1.0, x*0.01, x)
        features.eblc = features.eblc.str.upper()
        features.rename({'eblc': 'compression'}, axis=1, inplace=True)
        features = get_baselines_well(features) if 'solar' not in dataset else get_baselines_well_solar(features)
        features.set_index(['compression', 'eb'], inplace=True)
        model_results = all_models_results[all_models_results.data == dataset.upper()]
        model_results = model_results[model_results.te_metric == 'nrmse']
        model_results = model_results[['eb',  'te', 'TFE', 'compression', 'model']]
        joined_df = features.merge(model_results.reset_index(drop=True), on=['compression', 'eb']).drop_duplicates()
        joined_df['data'] = dataset
        all_datasets_features = pd.concat([all_datasets_features, joined_df])

    all_datasets_features.to_csv('../results/features/all_datasets_features_xg.csv', index=False)


def join_arima_pweather():
    df = pd.read_csv('../results/tfe/per_model/arima_results.csv')
    df = df[['error', 'nrmse', 'eblc', 'data', 'eb']]
    arima_results = df[df.data == 'weather']
    arima_results = get_tfe(arima_results)
    arima_results = arima_results[arima_results.eblc != 'baseline']
    arima_results.reset_index(drop=True, inplace=True)
    arima_results.eblc = arima_results.eblc.str.upper()
    arima_results['model'] = 'ARIMA'
    arima_results.data = arima_results.data.str.upper()
    arima_results['tfe_metric'] = 'nrmse'
    arima_results['te_metric'] = 'nrmse'
    arima_results['te'] = arima_results['error']
    x = arima_results.eb.values
    arima_results.eb = np.round(np.where(x >= 1.0, x*0.01, x), 3)
    arima_results = arima_results.rename({'nrmse': 'forecasting error', 'eblc': 'compression'}, axis=1)
    arima_results.drop('error', axis=1, inplace=True)
    pweather_results = pd.read_csv('../results/tfe/pprocessed_weather.csv')
    cr_df = pd.read_csv('../results/cr/pweather_cr.csv')
    cr_df.compression = cr_df.compression.str.upper()
    cr_df.data = cr_df.data.str.upper()
    cr_df.rename({'gzip': 'cr', 'error_bound': 'eb'}, axis=1, inplace=True)
    cr_df.set_index(['data', 'compression', 'eb'], inplace=True)
    joined = cr_df.merge(arima_results, on=['data', 'compression', 'eb'])
    concatenated = pd.concat([pweather_results, joined])
    concatenated['data'] = 'WEATHER'
    concatenated.to_csv('../results/tfe/pweather_results_.csv', index=False)


def plot_new_te_tfe_per_model(data_name, eb_bound):
    all_results = pd.read_csv('../results/tfe/models_results_xg_final.csv')
    all_results.eb = np.round(all_results.eb.values, 3)

    all_results = all_results[(all_results.data == data_name) & (all_results.eb <= eb_bound)
                              & (all_results.eb > 0) & (all_results.te_metric == 'nrmse')]

    if 'SOLAR' in data_name or 'AUS' in data_name:
        all_results.at[(all_results.model == 'GRU'), 'TFE'] = 0.0


if __name__ == '__main__':
    # compute_features_tfe_corr_xg()
    # join_with_arima()
    # join_arima_pweather()
    plot_new_te_tfe_per_model('WIND', 0.17)
