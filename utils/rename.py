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
