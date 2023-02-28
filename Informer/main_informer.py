import torch
import numpy as np
from Informer.exp.exp_informer import ExpInformer


def main_informer(args):
    data_parser = {
        'ettm1': {'data': 'ettm1.parquet', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ettm2': {'data': 'ettm2.parquet', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'solar': {'data': 'solar.parquet', 'T': '136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'weather': {'data': 'weather.parquet', 'T': 'OT', 'M': [22, 22, 22], 'S': [1, 1, 1], 'MS': [22, 22, 1]},
        'wind': {'data': 'wind.parquet', 'T': 'active_power', 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 1]},

    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)

    Exp = ExpInformer
    exp = None
    for ii in range(args.itr):
        for eb in args.EB:
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_eb{}_train_raw{}_{}_{}'.format(args.model_id,
                                                                                                                    args.data,
                                                                                                                    args.features,
                                                                                                                    args.seq_len,
                                                                                                                    args.label_len,
                                                                                                                    args.pred_len,
                                                                                                                    args.d_model,
                                                                                                                    args.n_heads,
                                                                                                                    args.e_layers,
                                                                                                                    args.d_layers,
                                                                                                                    args.d_ff,
                                                                                                                    args.attn,
                                                                                                                    args.factor,
                                                                                                                    args.embed,
                                                                                                                    args.distil,
                                                                                                                    args.mix,
                                                                                                                    eb,
                                                                                                                    args.train_raw,
                                                                                                                    args.des, ii)
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
        

