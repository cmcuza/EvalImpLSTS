import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join


def prepare(x, y=None):
    if "numpy" not in str(type(x)):
        x = np.array(x, dtype=np.float64)
    if "numpy" not in str(type(y)) and y is not None:
        y = np.array(y, dtype=np.float64)

    if x.shape[0] != y.shape[0]:
        new_x = np.zeros(y.shape, dtype=np.float64)
        fh = y.shape[1]
        for i in range(y.shape[0]):
            new_x[i] = x[i:i+fh]
        x = new_x

    if len(x.shape) > 1 and x.shape[1] != y.shape[1]:
        x = x[:, -y.shape[1]:]

    return x, y


def print_forecast(x, y, title='', args=None):
    x, y = prepare(x, y)

    if len(x.shape) == 2:
        start_point = x.shape[1]
        end_point = min(x.shape[0], start_point + 2 * x.shape[1])
        max_size = end_point - start_point
        original = np.empty(max_size)
        next_point_forecast = np.empty(max_size)
        middle_forecast = np.empty(max_size)
        long_forecast = np.empty(max_size)
        fh = x.shape[1]
        j = 0
        for i in range(start_point, end_point):
            original[j] = x[i][0]
            next_point_forecast[j] = y[i][0]
            middle_forecast[j] = y[i-fh//2][fh//2]
            long_forecast[j] = y[i-fh+1][-1]
            j += 1

            if x[i-fh//2][fh//2] != x[i][0] or x[i-fh+1][-1] != x[i][0]:
                raise Exception()

        plt.plot(original, label='original ts')
        plt.plot(next_point_forecast, label='next point forecast')
        plt.plot(middle_forecast, label='middle term forecast')
        plt.plot(long_forecast, label='long term forecast')
    else:
        i = min(1000, x.shape[0])
        plt.plot(x[:i], label='original')
        plt.plot(y[:i], label='forecast')

    plt.title(title)
    plt.legend()
    if args:
        exp_name = args.name
        data_name = args.data
    else:
        exp_name = '_'.join(title.split('_')[:-3])
        data_name = title.split('_')[-3]

    makedirs(join('..', 'figures', data_name, exp_name), exist_ok=True)
    plt.savefig(join('..', 'figures', data_name, exp_name, title + '.png'))
    plt.clf()
