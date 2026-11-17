import os

import torch
from torchvision import transforms,datasets,utils

import numpy as np 
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


"""---------数据加载及处理---------"""
#定义图片预处理的一套流程，在 image_dataset 读取时作为参数，依次对每一张照片进行处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'valid': transforms.Compose([
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
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                    for x in ['train','valid']}   # 这决定了dataset文件夹下路径是train/val分类为上一级

dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=4)
                    for x in ['train','valid']}
# 实际上是把前面创建的两个数据集对象交给 DataLoader;DataLoader：负责按照批次（Batch）读取数据

dataset_sizes = {x: len(image_datasets[x]) for x in ['train','valid']}
class_names = image_datasets['train'].classes  #从ImageFolder数据集中获取所有类别的名称列表

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""---------可视化---------"""
def imshow(inp,title = None):
    # inp 是传进来的 pytorch 的tensor类型
    inp = inp.numpy().transpose((1,2,0))
    # PyTorch 默认的图像维度顺序是 (通道数, 高度, 宽度)，而绘图库Matplotlib要求(高度, 宽度, 通道数)
    mean = np.array([0.485, 0.456, 0.406])  # 平均值
    std = np.array([0.229, 0.224, 0.225])   # 标准差
    inp = std * inp + mean  #去标准化
    inp = np.clip(inp, 0, 1)   # 防止浮点数计算误差，将还原后的像素值控制在 [0, 1] 范围
    plt.imshow(inp)  # 绘制图像
    if title is not None: 
        plt.title(title)  #绘制标题
    plt.pause(0.001)  # pause a bit so that plots are updated

inputs,classes = next(iter(dataloaders['train']))  # 取出dataloaders中的一个batch的图像和类别
    # next() 返回迭代器的下一个项目,next() 函数要和生成迭代器的iter()函数一起使用。
out = utils.make_grid(inputs)  # 把上面取出来的batch个照片拼接成一张网格图，方便一次性显示

imshow(out,title=[class_names[x] for x in classes])





"""---------定义CNN模型---------"""
class CNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )    # 224*224 -> 112*112
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )   # 112*112 ->56*56
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )   # 56*56 -> 28*28 , 此时x.shape = [4(batch_size), 64, 28, 28] ,卷积层输出四维tensor
        self.out = nn.Linear(64*28*28,2)
        """
        nn.Linear() 希望接收的是 [Batch数量, 特征数量] 的二维tensor
        out是全连接层,传参：输入的神经元个数(尺寸,in_features)、输出的神经元个数(类别数,out_features),bias=True # 是否包含偏置
        in_features 指的是输入的二维张量的大小，即输入的 [batch_size, size] 中的 size 。
        out_features 指的是输出的二维张量的大小，即输出的二维张量的形状为 [ batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
        从输入输出的张量的 shape 角度来理解，相当于一个输入为 [batch_size, in_features] 的张量变换成了 [batch_size, out_features] 的输出张量。
        """

    def forward(self,x):
        # x并不是一张普通的图片，而是已经经过预处理、转换成 Tensor 的图片数据。
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)  # 把[B,C,H,W]->[B,C*H*W],展平为二维tensor
        # torch.flatten(input, start_dim=0, end_dim=-1) 传参：输入张量,  开始、结束展平的维度
        out_put = self.out(x)
        return out_put


model_1 = CNN_1()  # 实例化模型
model_1 = model_1.to(device)    # 将模型移动到指定设备





"""---------训练模型---------"""
def train(train_loader,model,loss_fn,optimizer):
    # 数据加载器，神经网络模型，损失函数，优化器
    model.train()
    batch_num = 1   # 指示训练到第几个batch了  
    for inputs,labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        loss = loss_fn(pred,labels) #计算出模型预测的损失, loss是一个张量

        optimizer.zero_grad()   # 把上一批计算的梯度清理干净。
        loss.backward() # 根据预测结果的损失，逆向计算梯度
        optimizer.step()   # 根据梯度调整模型参数

        loss_value = loss.item()    # 获取损失参数;  .item()用于从张量中提取单个元素的值，并将其转换为Python基本类型
        print(f"{loss_value}  {batch_num}")
        batch_num += 1





"""---------定义测试函数---------"""
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss,correct = 0,0     # 记录累计损失和累计正确个数
    with torch.no_grad:  # 测试的时候不跟踪梯度
        for inputs,labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)
            loss = loss_fn(pred,labels)
            test_loss += loss.item() * inputs.size(0)   # inputs.size(0)是这个batch中有多少张图片
            correct += (pred.argmax(1)==labels).sum().item()
            # numpy.argmax() 函数可以返回最大值的索引，dim = 1表示沿下标为1的维度进行依个比较

        avg_loss = test_loss/size
        accuracy = correct/size

        print(f"Test result:\n Accuracy:{(accuracy*100)}%, Avg_loss:{(avg_loss*100)}%")


"""---------定义损失函数和优化器---------"""
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(),lr=0.001)


"""---------训练与测试模型---------"""
train(dataloaders['train'],CNN_1,loss_fn,optimizer)
test(dataloaders['valid'],CNN_1,loss_fn)


"""---------训练多个周期,然后测试---------"""
epochs = 50
for t in range(epochs):
    print(f"epoch {t+1}:\n")
    train(dataloaders['train'],CNN_1,loss_fn,optimizer)
print("done")
test(dataloaders['valid'],CNN_1,loss_fn)


