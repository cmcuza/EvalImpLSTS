import numpy as np


def R(x):
    return np.abs(x.max()-x.min())


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))


def NRMSE(pred, true):
    return RMSE(pred, true)/R(true)


def CORR(pred, true):
    true_m1 = true - true.mean(0)
    pred_m1 = pred - pred.mean(0)
    u = (true_m1*pred_m1).sum(0)
    d = np.sqrt((true_m1**2).sum(0)*(pred_m1**2).sum(0))
    return (u/d).mean()


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


def ACE(pred, true, lag=1):
    error = np.abs(pred-true)
    return np.corrcoef(np.array([error[:-lag], error[lag:]]))[0, 1]


def PSNR(pred, true):
    return 20*np.log10(R(true)/RMSE(pred, true))


def metrics(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    return mae, mse, rmse, mape, mspe, rse, corr
