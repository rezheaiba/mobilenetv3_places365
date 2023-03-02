"""
# @Time    : 2023/2/8 13:43
# @File    : dataloader.py
# @Author  : rezheaiba
"""
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class Loader(Dataset):
    def __init__(self, img_path, label_path, root, transforms_=None):
        self.name = []
        self.label1 = []
        self.label2 = []
        label_dict = {}
        with open(label_path, encoding='utf-8-sig') as f:
            for line in f.readlines():
                items = line.rstrip().split()
                label_dict[items[0]] = int(items[-1]) - 1
        with open(img_path, encoding='utf-8-sig') as f:
            for row in f.readlines():
                self.name.append(str(row.rstrip()))
                self.label1.append(int(label_dict[str(row).split('/')[-2]]))
                self.label2.append(int(5))
        self.root = root
        if transforms_ is not None:
            self.transform = transforms_

    def __getitem__(self, index):
        path = os.path.join(self.root, self.name[index])
        img = Image.open(path).convert("RGB")
        label1 = self.label1[index]
        label2 = self.label2[index]

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img,
                "label1": torch.tensor(label1, dtype=torch.int64),
                "label2": torch.tensor(label2, dtype=torch.int64),
                }

    def __len__(self):
        return len(self.name)


'''if __name__ == '__main__':
    dataloader = Loader(img_path=r'D:\Dataset\places365_standard\train.txt',
                        label_path=r'D:\Dataset\places365_standard\IO_places365.txt',
                        root='D:\Dataset\places365_standard',
                        transforms_=[transforms.Resize((224, 224)), transforms.ToTensor()],
                        )
    print(len(dataloader))
    data = DataLoader(dataloader,
                      batch_size=64,
                      shuffle=True,
                      num_workers=0)
    print(len(data))'''
