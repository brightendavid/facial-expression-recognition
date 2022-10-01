#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

'''df = pd.read_csv("C:/Users/brighten/Downloads/实验6 数据挖掘分类预测实验I/data-train.csv")
print(df.head())  # 打印head和前几个数据

print(df.loc[0])'''
# 遍历获取值，则：
'''for data in df.values:
    print(data)  # 列表'''


class MyDataset(Dataset):
    def __init__(self, csv_data):
        # self.data = pd.read_csv("C:/Users/brighten/Downloads/实验6 数据挖掘分类预测实验I/data-train.csv",header=None)
        self.data = pd.read_csv(csv_data, dtype='a')
        # self.label = np.array(self.data['emotion'])
        self.img_data = np.array(self.data['pixels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.img_data[idx]
        x = np.fromstring(x, dtype=float, sep=' ')
        # x = x / x.max()

        label = [0, 0, 0, 0, 0, 0, 0]
        # y = int(self.label[idx])
        y=0
        label[y] = 1

        # x /= np.max(x) #不明原因，加入归一化，loss反而变大 ，可能归一化会把像素之间的差别减小，导致了最终的结果忽略了这个变化
        x = torch.from_numpy(x)
        label = np.array(label)
        label = torch.from_numpy(label)

        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return x, label


# 用法
if __name__ == "__main__":
    s = MyDataset("./test_data.csv")
    print(s.__len__())
    # print(s.__getitem__(3))
    data, label = s.__getitem__(3)
    print(data)
    print(data.shape)
    print(label)
    print(int(label[5]))
