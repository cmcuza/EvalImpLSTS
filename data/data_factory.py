import os
from data.data_loader import ETT, Solar, Wind, Weather, AUSElecDem

from torch.utils.data import DataLoader

data_dict = {
    'ettm1': ETT,
    'ettm2': ETT,  # SZ compression
    'solar': Solar,
    'weather': Weather,
    'wind': Wind,
    'aus': AUSElecDem
}


def data_provider(args, flag):
    Data = data_dict[args.data.split('_')[0]]

    if flag in ['test', 'pred']:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path+os.path.sep+args.eblc,
        data=args.data,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target_var,
        freq=freq,
        eb=args.eb,
        retrain=args.retrain
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
