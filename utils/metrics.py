import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))


def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred-true))


def SMAPE(y, p):
    return np.mean(2 * np.abs(np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-8)))


def MSE(pred, true):
    return np.mean((pred-true)**2)


def MSMAPE(y, p):
    epsilon = 0.1
    comparator = np.repeat((0.5 + epsilon), repeats=np.prod(y.shape)).reshape(y.shape)
    den = np.maximum(comparator, (np.abs(p) + np.abs(y) + epsilon))
    smape = 2 * np.abs(y - p) / den
    return np.mean(smape)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    return mae, mse, rmse, mape, mspe, rse, corr
