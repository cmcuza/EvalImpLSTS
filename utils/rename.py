import os
import argparse


def rename(root_path, old, new, re=''):
    if re == '':
        re = old
    count = 0
    for root, dr, files in os.walk(root_path):
        for drs in dr:
            if drs.find(re) != -1:
                if drs.find('copy') != -1:
                     drs = drs.replace(' ', '\ ')

                newf = drs.replace(old, new)
                os.system(f'mv {os.path.join(root_path, drs)} {os.path.join(root, newf)}')
                # print(f'mv {os.path.join(root_path, drs)} {os.path.join(root, newf)}')
                count += 1
    print(count, 'files changed')


# import pickle as pkl

# def fix_wind():
#     for root, dr, files in os.walk('output/arima/wind/'):
#         for file in files:
#             if 'output' in file:
#                 with open(os.path.join(root, file), 'rb') as f:
#                     out = pkl.load(f)
#                 out = out[1:]
#                 with open(os.path.join(root, file), 'wb') as f:
#                     pkl.dump(out, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='root dir')
    parser.add_argument('--re', help='regular expression to match', default='')
    parser.add_argument('--old', help='expression to change')
    parser.add_argument('--new', help='expression to change')
    args = parser.parse_args()
    
    root_path = args.root
    re = args.re
    old = args.old
    new = args.new
    
    rename(root_path, old, new, re)
