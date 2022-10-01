#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
from autoencoder import AutoEncoder
import torch.nn as nn
import torchvision
import numpy as np

import data2
import net

# 定义一些超参数
# batch_size = 64
import dataloader

learning_rate = 0.0001
# num_epoches = 20

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# 数据集的下载器
'''train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)'''
train_loader = data2.MyDataset("./train_data.csv")
test_loader = dataloader.MyDataset("./test_data.csv")
# 选择模型
# model = net.simpleNet(28 * 28, 300, 100, 10)
# model = net.Activation_Net(28 * 28, 300, 100, 10)
model = net.Activation_Net(1 * 2304, 1024, 512, 256, 7)
# model=net.Batch_Net(1*2304,1024,512,7)
'''if torch.cuda.is_available():
    model = model.cuda()'''

# 定义损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# 训练模型
los = []
epoch_item = []


def train():
    epoch = 0
    for i in range(100):
        for data in train_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
            '''if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:'''
            img = Variable(img)
            label = Variable(label)
            out = model(img)
            # print(out.shape)
            # print(label.shape)
            loss = criterion(out, label)
            print_loss = loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1
            if epoch % 50 == 0:
                print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
                los.append(loss.data.item())
                epoch_item.append(epoch)

    torch.save(model.state_dict(), 'netpig.pt')
    # 记录loss
    plt.title("square of 'x'", fontsize=20)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.ylim(0, 2)
    plt.plot(epoch_item, los)
    # 设置坐标轴刻度标记的大小
    plt.tick_params(axis='both', labelsize=10)
    plt.savefig("1.png")


def test_the_model1():
    # 模型评估
    acc = 0
    emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    # test 部分有问题
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    model = net.Activation_Net(1 * 2304, 1024, 512, 256, 7)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 加载模型参数
    model.load_state_dict(torch.load('./netpig.pt', map_location=device))
    # 测试模式
    model.eval()
    ss = []
    eval_loss = 0
    eval_acc = 0
    for data in train_loader:
        img, label = data
        '''print(img)
        print(label)'''
        # img = img.view(img.size(0), -1)
        ''' if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()'''

        '''    out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
        '''
        img = Variable(img)
        label = Variable(label)
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        '''print(out.shape)
            print(label.shape)'''
        loss = criterion(out, label)
        # eval_loss += loss.data.item() * label.size(0)
        # print(label.size(0)) =  2
        '''if not isinstance(out, list):
            out = [out]'''
        print("out")
        print(out)
        # _, pred = torch.max(out, 1)

        z = int(torch.argmax(out))
        print(z)
        print(label)
        if int(label[z]) == 1:
            acc += 1

        pred = emotion[z]
        print(pred)
        # ss.append(pred)
        # ss.append(z)

    print('Acc: {:.6f}%'.format((acc / train_loader.__len__()) * 100.0))


def test_the_model():
    # 模型评估
    acc = 0
    emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    # test 部分有问题
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    model = net.Activation_Net(1 * 2304, 1024, 512, 256, 7)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 加载模型参数
    model.load_state_dict(torch.load('./netpig.pt', map_location=device))
    # 测试模式
    model.eval()
    ss = []
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        '''print(img)
        print(label)'''
        # img = img.view(img.size(0), -1)
        ''' if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()'''

        '''    out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
        '''
        img = Variable(img)
        label = Variable(label)
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        '''print(out.shape)
            print(label.shape)'''
        loss = criterion(out, label)
        # eval_loss += loss.data.item() * label.size(0)
        # print(label.size(0)) =  2
        '''if not isinstance(out, list):
            out = [out]'''
        print("out")
        print(out)
        # _, pred = torch.max(out, 1)

        z = int(torch.argmax(out))
        print(z)
        if int(label[z]) == 1:
            acc += 1

        pred = emotion[z]
        print(pred)
        # ss.append(pred)
        ss.append(z)
        '''num_correct = (pred == label).sum()
        eval_acc += num_correct.item()'''

    '''print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / 223,
        eval_acc / 223
    ))
    '''

    with open('./result.txt', 'w') as f1:
        for i in ss:
            f1.write(str(i))
            f1.write("\n")
    # print('Acc: {:.6f}%'.format((acc / train_loader.__len__())*100.0))


if __name__ == "__main__":
    train()
    test_the_model1()
    # test_the_model()
