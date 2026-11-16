import os

import torch
from torchvision import transforms,datasets

#定义图片预处理的一套流程，在 image_dataset 读取时作为参数，依次对每一张照片进行处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) ,
}
# transforms.Compose的核心作用是将多个图像预处理或数据增强操作串联成一个按顺序执行的“流水线”


data_dir = 'dataset'

#分别读取训练集（train）和验证集（val），并将它们封装成PyTorch可以使用的图像数据集对象，最后存储在一个字典中。
#image_datasets 是由 ImageFolder 创建的数据集对象
image_dataset = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                    for x in ['train','val']}   # 这决定了dataset文件夹下路径是train/val分类为上一级

dataloaders = {x:torch.utils.data.DataLoader(image_dataset[x],batch_size=4,shuffle=True,num_workers=4)
                    for x in ['train','val']}
# 实际上是把前面创建的两个数据集对象交给 DataLoader;DataLoader：负责按照批次（Batch）读取数据

