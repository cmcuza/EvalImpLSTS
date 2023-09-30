import torch
from forecasting.DLinear.exp.exp_dlinear import LinearExp
import numpy as np


def main_dlinear(args):
    Exp = LinearExp
    exp = None
    for ii in range(args.itr):
        for eb in args.EB:
            args.eb = eb
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_{}_it_{}_eb_{}'.format(
                    args.model_id,
                    args.dataset,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.des, ii, eb) + '_retrain' if args.retrain == 1 else ''

            if eb == 0.0:
                exp = Exp(args)  # set experiments

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            else:
                for eblc in ['pmc', 'swing', 'sz']:
                    args.eblc = eblc
                    if eblc == 'sz':
                        eb = np.round(eb * 0.01, 4)
                    if args.retrain == 0:
                        print('>>>>>>>predicting eb_{}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eb, setting))
                        exp.predict(setting, eb=eb)
                    else:
                        exp.is_retrain = True
                        print('>>>>>>>retraining eb_{}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eb, setting))
                        exp.retrain(setting, eb=eb)
                        print('>>>>>>>predicting eb_{}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eb, setting))
                        exp.predict(setting, eb=eb)

        torch.cuda.empty_cache()

