#!/usr/bin/env python
# coding: utf-8

import os, argparse, shutil

def main():
    parser = argparse.ArgumentParser(description='testsh')
    parser.add_argument('--dir', type=str, default='result/')
    args = parser.parse_args()

    goals = []
    for folder2 in os.listdir(args.dir):
        folder2_path = os.path.join(args.dir, folder2)
        if os.path.isfile(folder2_path): # ファイルだったらスキップ
                continue
        for folder in os.listdir(folder2_path):
            folder_path = os.path.join(folder2_path, folder)
            if os.path.isfile(folder_path): # ファイルだったらスキップ
                continue
            if not os.path.isfile(os.path.join(folder_path, 'train.csv')): # train.csvがない(学習完了していない)ならスキップ
                continue
            if os.path.isdir(os.path.join(folder_path, 'ad')): # すでに異常検知済みならスキップ
                continue
            goals.append(folder_path)

    goals.sort()
    with open('test.sh', 'w') as f:
        f.write('#!/bin/sh\necho "Start"\n')
        for goal in goals:
            f.write('python3 test.py --save_dir "{}"\n'.format(goal))
            print('python3 test.py --save_dir "{}"'.format(goal))

if __name__ == '__main__':
    main()
