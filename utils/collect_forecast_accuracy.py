import numpy as np
import os
from os.path import sep
import pandas as pd
import re
import pickle as pkl
from utils.metrics import *


output_path = os.path.join('..', 'outputs', 'predictions')


def metrics_ensemble(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    rse = RSE(pred, true)
    nrmse = NRMSE(pred, true)
    corr = CORR(pred, true)
    psnr = PSNR(pred, true)

    return {'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'rse': rse,
            'corr': corr,
            'psnr': psnr}


def load_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def is_valid(exp, dataset, model, eblc):
    d = exp.find(dataset)
    m = exp.find(model)
    e = exp.find(eblc)
    o = exp.find('output')

    return d+m+e+o == -4


def get_prediction_metrics(raw_ts, dataset, model, eblc):
    metrics = []
    ebs = []
    for root, dr, file in os.walk(output_path):
        for exp in file:
            if is_valid(exp, dataset, model, eblc):
                results = metrics_ensemble(load_pkl(output_path + sep + exp), raw_ts)
                metrics.append(results)
                ebs.append(float(re.findall('eb_0.[0-9]+', exp)[0][3:]))

    forecasting_metrics = pd.DataFrame(metrics)
    forecasting_metrics['eb'] = ebs
    forecasting_metrics = forecasting_metrics.groupby(['eb']).median()

    return forecasting_metrics
