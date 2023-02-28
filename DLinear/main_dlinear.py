import torch
from DLinear.exp.exp_dlinear import ExpMain
import numpy as np


def main_dlinear(args):
    Exp = ExpMain
    exp = None
    for ii in range(args.itr):
        for eb in args.EB:
            args.eb = eb
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_{}_it_{}_eb_{}'.format(
                    args.model_id,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.des, ii, eb)

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
                    print('>>>>>>>predicting eb_{}: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eb, setting))
                    exp.predict(setting, eb=eb)

        torch.cuda.empty_cache()

