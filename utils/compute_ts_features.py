import os
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.0.3'
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import FloatVector, StrVector
# from config.config import SEASONALITY_MAP, FEATURES_NAME
from data.data_loader import ETT, Wind, Weather, Solar, AUSElecDem

SEASONALITY_MAP = {'ettm1': 96, 'ettm2': 96, 'aus': 48, 'solar': 144, 'pweather': 144, 'wind': 144}
FEATURES_NAME = ("max_kl_shift", "max_level_shift", "max_var_shift", "acf_features", "arch_stat", "crossing_points",
                 "entropy", "flat_spots", "holt_parameters", "hurst", "lumpiness", "nonlinearity", "pacf_features",
                 "stability", "unitroot_kpss", "unitroot_pp")


def get_test_df(data, fixed_borders):
    if fixed_borders:
        border1s = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - 96
        border2s = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
        # border1s = [0, 6000 - self.seq_len, 8000 - self.seq_len]
        # border2s = [6000, 8000, 10000]
    else:
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        border1s = len(data) - num_test - 96
        border2s = len(data)

    test_data = data[border1s:border2s]

    return test_data


def get_features(data_name, eblc, target_var) -> pd.DataFrame:
    data = pd.read_parquet(f'../data/compressed/{eblc}/{data_name}')
    bfix = True if 'ett' in data_name else False
    test_series_data = data[target_var]
    if 'test' not in data_name:
        test_series_data = get_test_df(test_series_data, bfix)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(test_series_data)

    seasonality = SEASONALITY_MAP[data_name.split('_')[1]]

    robjects.r.assign("seasonality", FloatVector([seasonality]))
    robjects.r.assign("TSFEATURE_NAMES", StrVector(FEATURES_NAME))
    robjects.r.assign("series_data", r_from_pd_df)
    robjects.r('library(dplyr)')
    robjects.r('library(tidyverse)')
    robjects.r('library(Rcatch22)')

    r_script = """
    start_date <- start(as.ts(series_data, frequency = max(seasonality)))

    if (length(start_date) == 1){  
        start_date <- c(floor(start_date), floor((start_date - floor(start_date)) * max(seasonality)))
    }

    series <- forecast:::msts(series_data, start=start_date, seasonal.periods = seasonality, ts.frequency = floor(max(seasonality)))
    features <- tsfeatures:::tsfeatures(series, c("mean","var"), scale = FALSE, na.rm = TRUE)
    
    for(f in TSFEATURE_NAMES){
        calculated_features <- tsfeatures:::tsfeatures(series, features = f)

        if(sum(is.na(calculated_features)) > 0){ 
            calculated_features <- tsfeatures:::tsfeatures(ts(series, frequency = 1), features = f)

            if(sum(is.na(calculated_features)) > 0){ 
                if(f == "max_kl_shift" | f == "max_level_shift" | f == "max_var_shift")
                calculated_features <- tsfeatures:::tsfeatures(series, features = f, width = 1)
                else{
                    if(f == "arch_stat")
                        calculated_features <- tsfeatures:::tsfeatures(series, features = f, lag = 1)
                }
            }
        }

        features <- bind_cols(features, calculated_features)

    }

    # Calculating stl_features
    tryCatch( 
        seasonal_features <- tsfeatures:::tsfeatures(series,"stl_features", s.window = 'periodic', robust = TRUE)
        , error = function(e) {
          tryCatch({
            seasonal_features <<- tsfeatures:::tsfeatures(series,"stl_features")
          }, error = function(e) {
            seasonal_features <<- tsfeatures:::tsfeatures(ts(series, frequency = 1),"stl_features") # Ignoring seasonality
          })
        })
    

    features <- bind_cols(features, seasonal_features)
    
    """

    # features22 < - catch22_all(series)
    # features22 < - spread(features22, key=names, value=values)
    # features <- bind_cols(features, features22)

    robjects.r(r_script)
    features = robjects.r['features']
    features = pandas2ri.rpy2py(features)
    return features


def do_pweather():
    data = 'test_pweather'
    features_results = get_features(f'{data}_points.parquet', 'pmc', 'OT-R')
    features_results['eb'] = 0.0
    features_results['eblc'] = 'raw'
    for eblc in ['sz', 'pmc', 'swing']:
        for eb in [1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 2.5, 2.7, 3.0, 5.0, 7.0, 10.0]:

            if 'sz' in eblc:
                eb *= 0.01
                eb = round(eb, 5)

            ot = f'OT-E{eb}'
            print(eblc, eb)
            eb_results = get_features(f'{data}_points.parquet', eblc, ot)
            eb_results['eb'] = eb
            eb_results['eblc'] = eblc
            features_results = pd.concat([features_results, eb_results])

    features_results['data'] = data
    features_results.to_csv(f'../results/features/{data}_features.csv', index=False)


def do_wind():
    data = 'test_wind'
    features_results = get_features(f'{data}_output_data_points.parquet', 'pmc', 'active_power-R')
    features_results['eb'] = 0.0
    features_results['eblc'] = 'raw'
    for eblc in ['sz', 'pmc', 'swing']:
        for eb in [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80]:
            print(eblc, eb)
            ot = f'active_power-E{eb}'
            if 'sz' in eblc:
                eb *= 0.01
                if 'wind' in data:
                    ot = f'active power-E{eb}'
            eb_results = get_features(f'{data}_output_data_points.parquet', eblc, ot)
            eb_results['eb'] = eb
            eb_results['eblc'] = eblc
            features_results = pd.concat([features_results, eb_results])

    features_results['data'] = data
    features_results.to_csv(f'../results/features/{data}_features.csv', index=False)


def do_etts():
    data = 'test_ettm1'
    features_results = get_features(f'{data}_output_data_points.parquet', 'pmc', 'OT-R')
    features_results['eb'] = 0.0
    features_results['eblc'] = 'raw'

    for eblc in ['sz', 'pmc', 'swing']:
        for eb in [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80]:
            print(eblc, eb)
            ot = f'OT-E{eb}'
            if 'sz' in eblc:
                eb *= 0.01
            eb_results = get_features(f'{data}_output_data_points.parquet', eblc, ot)
            eb_results['eb'] = eb
            eb_results['eblc'] = eblc
            features_results = pd.concat([features_results, eb_results])

    features_results['data'] = data
    features_results.to_csv(f'../results/features/{data}_features.csv', index=False)


def do_aus():
    data = 'test_aus_electrical_demand'
    features_results = get_features(f'{data}_points.parquet', 'pmc', 'y-R')
    features_results['eb'] = 0.0
    features_results['eblc'] = 'raw'
    for eblc in ['sz', 'pmc', 'swing']:
        for eb in [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 65.0, 80.0]:
            if 'sz' in eblc:
                eb *= 0.01

            print(eblc, eb)
            ot = f'y-E{eb}'

            eb_results = get_features(f'{data}_points.parquet', eblc, ot)
            eb_results['eb'] = eb
            eb_results['eblc'] = eblc
            features_results = pd.concat([features_results, eb_results])

    features_results['data'] = data
    features_results.to_csv(f'../results/features/{data}_features.csv', index=False)


def do_solar():
    data = 'test_solar'
    # solar = pd.read_parquet(f'../data/compressed/pmc/{data}_output_data_points.parquet')
    all_var_results = pd.DataFrame()
    for var in range(137):
        features_results = get_features(f'{data}_output_data_points.parquet', 'pmc', f'{var}-R')
        features_results['eb'] = 0.0
        features_results['eblc'] = 'raw'
        for eblc in ['sz', 'pmc', 'swing']:
            for eb in [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80]:
                if 'sz' in eblc:
                    eb *= 0.01

                print(var, eblc, eb)
                ot = f'{var}-E{eb}'

                eb_results = get_features(f'{data}_output_data_points.parquet', eblc, ot)
                eb_results['eb'] = eb
                eb_results['eblc'] = eblc
                features_results = pd.concat([features_results, eb_results])
        features_results['var'] = var
        all_var_results = pd.concat([all_var_results, features_results])

    all_var_results = all_var_results.groupby(['eblc', 'eb']).mean().reset_index()
    all_var_results['data'] = data
    all_var_results.to_csv(f'../results/features/{data}_features.csv', index=False)


if __name__ == '__main__':
    do_solar()
