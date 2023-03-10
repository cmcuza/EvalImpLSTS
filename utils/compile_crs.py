from os.path import join
import numpy as np
import json
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


def compute_cr(raw, compressed, as_percentage=True):
    if as_percentage:
        return (1 - np.round(compressed / raw, 2)) * 100

    return np.round(raw / compressed, 2)


ettm1_pmc = json.load(open(join('..', 'results', 'pmc_ETTm1_cr.json')))
ettm2_pmc = json.load(open(join('..', 'results', 'pmc_ETTm2_cr.json')))
weather_pmc = json.load(open(join('..', 'results', 'pmc_weather_cr_big_eb.json')))
pweather_pmc = json.load(open(join('..', 'results', 'pmc_pweather_cr.json')))
belhle02_pmc = json.load(open(join('..', 'results', 'pmc_wind.json')))
solar_pmc = json.load(open(join('..','results', 'pmc_solar_cr.json')))
ettm1_swing = json.load(open(join('..', 'results', 'swing_ETTm1_cr.json')))
ettm2_swing = json.load(open(join('..', 'results', 'swing_ETTm2_cr.json')))
weather_swing = json.load(open(join('..', 'results', 'swing_weather_cr_big_eb.json')))
pweather_swing = json.load(open(join('..', 'results', 'swing_pweather_cr.json')))
belhle02_swing = json.load(open(join('..', 'results', 'swing_wind.json')))
solar_swing = json.load(open(join('..', 'results', 'swing_solar_cr.json')))
ettm1_sz = json.load(open(join('..', 'resullts', 'sz_ETTm1_cr.json')))
ettm2_sz = json.load(open(join('..', 'resullts', 'sz_ETTm2_cr.json')))
weather_sz = json.load(open(join('..', 'resullts', 'sz_weather_cr_big_eb.json')))
solar_sz = json.load(open(join('..', 'resullts', 'sz_solar_cr.json')))
belhle02_sz = json.load(open(join('..', 'resullts', 'sz_wind.json')))
weather_scaled_pmc = json.load(open(join('..', 'resullts', 'pmc_weather_scaled_cr.json')))
weather_scaled_swing = json.load(open(join('..', 'resullts', 'swing_weather_scaled_cr.json')))
solar_gorillas = json.load(open(join('..', 'resullts', 'gorillas_solar_cr.json')))
belhle02_gorillas = json.load(open(join('..', 'resullts', 'gorillas_wind.json')))
ettm1_gorillas = json.load(open(join('..', 'resullts', 'gorillas_ETTm1_cr.json')))
ettm2_gorillas = json.load(open(join('..', 'resullts', 'gorillas_ETTm2_cr.json')))
weather_gorillas = json.load(open(join('..', 'resullts', 'gorillas_weather_cr.json')))

list_cr = []

for data, name in [(ettm1_pmc, 'ettm1_pmc'),
                   (ettm1_swing, 'ettm1_swing'),
                   (ettm2_pmc, 'ettm2_pmc'),
                   (ettm2_swing, 'ettm2_swing'),
                   (weather_pmc, 'weather_pmc'),
                   # (pweather_pmc, 'pweather_pmc'),
                   (weather_swing, 'weather_swing'),
                   # (pweather_swing, 'pweather_swing'),
                   (ettm1_gorillas, 'ettm1_gorillas'),
                   (ettm2_gorillas, 'ettm2_gorillas'),
                   (weather_gorillas, 'weather_gorillas')]:

    uncompressed = data['OT-R']['segments']['gzip']
    df_ratio = pd.DataFrame()

    for k, v in data.items():
        lossy_ratio = {}
        for key, value in data[k]['segments'].items():
            lossy_ratio[key] = [compute_cr(uncompressed, value)]
        df_lossy = pd.DataFrame.from_dict(lossy_ratio)
        df_lossy['error_bound'] = int(k[4:]) * 0.01 if len(k) > 4 else 0
        df_ratio = pd.concat([df_lossy, df_ratio])

    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(pweather_pmc, 'pweather_pmc'),
                   (pweather_swing, 'pweather_swing')]:

    uncompressed = data['OT-R']['segments']['gzip']
    df_ratio = pd.DataFrame()

    for k, v in data.items():
        lossy_ratio = {}
        for key, value in data[k]['segments'].items():
            lossy_ratio[key] = [compute_cr(uncompressed, value)]
        df_lossy = pd.DataFrame.from_dict(lossy_ratio)
        df_lossy['error_bound'] = np.round(float(k[4:]) * 0.01, 4) if len(k) > 4 else 0.0
        df_ratio = pd.concat([df_lossy, df_ratio])

    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(ettm1_sz, 'ettm1_sz'),
                   (ettm2_sz, 'ettm2_sz'),
                   (weather_sz, 'weather_sz')]:

    uncompressed = data['OT-R']
    df_ratio = pd.DataFrame()
    lossy_ratio = []
    error_bound = []
    for k, v in data.items():
        lossy_ratio.append(compute_cr(uncompressed, v))
        error_bound.append(float(k[1:]) if k.startswith('E') else 0)

    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

#
#
for data, name in [(solar_sz, 'solar_sz')]:
    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        counter_vb.update(values)

    uncompressed = counter_vb['gzip']
    lossy_ratio = [1.0]
    error_bound = [0.0]
    for k, v in counter_vb.items():
        if k.startswith('E'):
            lossy_ratio.append(compute_cr(uncompressed, v))
            error_bound.append(float(k[1:]))

    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(belhle02_sz, 'belhle02_sz')]:
    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        if column.find('active power') != -1:
            counter_vb.update(values)

    uncompressed = counter_vb['gzip']
    lossy_ratio = [1.0]
    error_bound = [0.0]
    for k, v in counter_vb.items():
        if k.startswith('E'):
            lossy_ratio.append(compute_cr(uncompressed, v))
            error_bound.append(float(k[1:]))

    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(solar_pmc, 'solar_pmc'),
                   (solar_swing, 'solar_swing')]:

    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        c = column.split('-')[1]

        error_bound = int(c[1:] if c.startswith('E') else 0) * 0.01
        counter_vb.update({error_bound: values['segments']['gzip']})

    uncompressed = counter_vb[0.0]
    lossy_ratio = []
    error_bound = []
    for k, v in counter_vb.items():
        lossy_ratio.append(compute_cr(uncompressed, v))
        error_bound.append(k)
    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(solar_gorillas, 'solar_gorillas')]:

    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        c = column.split('-')[1]

        error_bound = int(1 if c.startswith('E') else 0) * 0.01
        counter_vb.update({error_bound: values['segments']['gzip']})

    uncompressed = counter_vb[0.0]
    lossy_ratio = []
    error_bound = []
    for k, v in counter_vb.items():
        lossy_ratio.append(compute_cr(uncompressed, v))
        error_bound.append(0.0)
    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(belhle02_pmc, 'belhle02_pmc'),
                   (belhle02_swing, 'belhle02_swing')]:

    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        if column.split('-')[0] == 'active_power':
            c = column.split('-')[1]
            error_bound = int(c[1:] if c.startswith('E') else 0) * 0.01
            counter_vb.update({error_bound: values['segments']['gzip']})

    uncompressed = counter_vb[0.0]
    lossy_ratio = []
    error_bound = []
    for k, v in counter_vb.items():
        lossy_ratio.append(compute_cr(uncompressed, v))
        error_bound.append(k)
    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

for data, name in [(belhle02_gorillas, 'belhle02_gorillas')]:

    df_ratio = pd.DataFrame()
    counter_vb = Counter()

    for column, values in data.items():
        if column.split('-')[0] == 'active_power':
            c = column.split('-')[1]
            error_bound = int(1 if c.startswith('E') else 0) * 0.01
            counter_vb.update({error_bound: values['segments']['gzip']})

    uncompressed = counter_vb[0.0]
    lossy_ratio = []
    error_bound = []
    for k, v in counter_vb.items():
        lossy_ratio.append(compute_cr(uncompressed, v))
        error_bound.append(0.0)
    df_ratio = pd.DataFrame(zip(lossy_ratio, error_bound), columns=['gzip', 'error_bound'])
    df_ratio['compression'] = name.split('_')[1]
    df_ratio['data'] = name.split('_')[0]
    list_cr.append(df_ratio)

concat_df = pd.concat(list_cr)

concat_df = concat_df.sort_values('error_bound')
concat_df.reset_index(inplace=True, drop=True)
concat_df.to_csv(join('..', 'resullts', 'cr', 'all_cr.csv'), index=None)

plt.rcParams['figure.figsize'] = [16, 12]
ettm1_cr = concat_df[concat_df['data'] == 'ettm1']
ettm2_cr = concat_df[concat_df['data'] == 'ettm2']
solar_cr = concat_df[concat_df['data'] == 'solar']
weather_cr = concat_df[concat_df['data'] == 'weather']
wind_cr = concat_df[concat_df['data'] == 'belhle02']


for i, df in enumerate([ettm1_cr, ettm2_cr, solar_cr, weather_cr, wind_cr]):
    plt.subplot(2, 3, i+1)
    if df['data'].iloc[0] == 'weather':
        df = df[df['error_bound'] < 0.1]
    plt.title(df['data'].iloc[0])
    sns.lineplot(data=df,  x='error_bound', y="gzip", hue='compression')