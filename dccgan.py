#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
# @Author : Luyouqi
# @License :
# @Contact : real_luyouqi@163.com
# @Software: PyCharm
# @File : dccgan.py
# @Time : 2022/8/28 21:06
# @Desc : 条件DCGAN
from collections import OrderedDict
import torch
from torch.nn import Linear, LeakyReLU, Sequential, BatchNorm1d, ReLU, Tanh
from torch.nn import BCELoss, MSELoss
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, MaxPool2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def conv_cat(x, y):
    """
    拼接特征图与标签，以便将标签加到输入中
    :param x:特征图 x.shape=b,c,h,w
    :param y:标签 y.shape=b,1
    :return:
    """
    batch_size = x.shape[0]
    # 对于全连接层使用全1数组拼接标签至特征列中
    if len(x.shape) == 2:
        y = torch.reshape(y, shape=(batch_size, 1))  # 确保标签值的形状
        one = torch.ones_like(y, dtype=torch.float)  # 图象数据将标签加在了通道上,2d数据将标签加在了特征上
        # y.shape=b,1
        # one.shape=b,1
        # (y*one).shape=b,1
        # 将y拼接到x之后,x增加了1列作为特征,这要求x是(b,n)的形状
    # 对于4D特征图将标签作为一个Channel拼接至特征图中
    elif len(x.shape) == 4:
        y = torch.reshape(y, shape=(batch_size, 1, 1, 1))  # y.shape=b,1,1,1
        one = torch.ones_like(x, dtype=torch.float)  # 图象数据将标签加在了通道上,2d数据将标签加在了特征上
        # one.shape=b,1,28,28
        # y.shape=b,1,1,1  对y和one做点乘,触发自动广播机制
        # (y*one).shape=b,1,28,28
        # x.shape=b,2*c,28,28
        # 每将y拼接到x上一次，x的第1个维度(维度从0开始)就会翻倍
    x = torch.cat((x, y * one), dim=1)
    return x


class Dist(torch.nn.Module):
    """
    判别器
    """

    def __init__(self):
        super(Dist, self).__init__()
        # 第一组卷积层
        self.cnn1 = Sequential(OrderedDict([
            ("conv1", Conv2d(in_channels=1 + 1, out_channels=64, kernel_size=(3, 3))),
            ("bn1", BatchNorm2d(num_features=64)),
            ('ac1', LeakyReLU()),
            ('pl1', MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        ]))
        # 第二组卷积层
        self.cnn2 = Sequential(OrderedDict([
            ('conv2', Conv2d(in_channels=64 + 64, out_channels=128, kernel_size=(3, 3))),
            ('bn2', BatchNorm2d(num_features=128)),
            ('ac2', LeakyReLU()),
            ('pl2', MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        ]))
        # 全连接层
        self.fc1 = Sequential(OrderedDict([
            ('ln1', Linear(in_features=128 * 5 * 5 + 1, out_features=1024)),
            ('bn1', BatchNorm1d(num_features=1024)),
            ('ln2', Linear(in_features=1024, out_features=1)),
            # ('ac1', Sigmoid())  # 最后别忘了加上Sigmoid将输出规整化到0-1范围内
        ]))

    def forward(self, x, y):
        """
        前向传递函数
        :param x: 图象数据 x.shape=b,1,28,28
        :param y: 图象标签 y.shape=b,1
        :return:
        """
        batch_size = x.shape[0]
        x = conv_cat(x, y)
        o1 = self.cnn1(x)
        o1 = conv_cat(o1, y)
        o2 = self.cnn2(o1)
        o2 = torch.reshape(o2, shape=(batch_size, -1))  # shape=128,3200
        o2 = conv_cat(o2, y)  # shape=b,6400
        o3 = self.fc1(o2)  # shape=b,1
        return o3


class Gene(torch.nn.Module):
    """
    生成器
    """

    def __init__(self):
        super(Gene, self).__init__()
        # 第一组全连接层
        self.fc1 = Sequential(OrderedDict([
            ('ln1', Linear(in_features=100 + 1, out_features=1024)),
            ('bn1', BatchNorm1d(num_features=1024)),
            ('ac1', ReLU())
        ]))
        # 第二组全连接层
        self.fc2 = Sequential(OrderedDict([
            ('ln2', Linear(in_features=1024 + 1, out_features=128 * 7 * 7)),
            ('bn2', BatchNorm1d(num_features=128 * 7 * 7)),
            ('ac2', ReLU())
        ]))
        # 第一组转置卷积
        self.transcnn1 = Sequential(OrderedDict([
            ('tconv1',
             ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ('bn1', BatchNorm2d(num_features=64)),
            ('ac1', ReLU())
        ]))
        # 第二组反卷积
        self.transcnn2 = Sequential(OrderedDict([
            ('tconv2', ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=1)),
            ('ac2', ReLU())  # 原文使用ReLU，这里使用Tanh使像素值在-1-1之间
        ]))

    def forward(self, z, label):
        """
        :param z: 噪声数据,shape=b,100,1,1
        :param label: 标签数据,shape=b,1
        :return:
        """
        batch_size = z.shape[0]
        z = torch.reshape(z, shape=(batch_size, -1))
        z = conv_cat(z, label)
        o1 = self.fc1(z)
        o1 = conv_cat(o1, label)
        o2 = self.fc2(o1)  # shape=b,6272
        o2 = torch.reshape(o2, shape=(batch_size, 128, 7, 7))  # b,128,7,7
        o2 = conv_cat(o2, label)  # b,256,7,7
        o3 = self.transcnn1(o2)
        o3 = conv_cat(o3, label)
        o4 = self.transcnn2(o3)
        return o4


class ConditionalGAN:
    """
    条件GAN
    """

    def __init__(self, lr=2e-4):
        super(ConditionalGAN, self).__init__()
        # 实例化生成器与判别器
        self.gene = Gene()
        self.dist = Dist()
        # 设置优化器
        self.opti_gene = Adam(lr=lr, params=self.gene.parameters())
        self.opti_dist = Adam(lr=lr, params=self.dist.parameters())
        # 设置损失函数
        self.loss_dist = MSELoss()  # 判别器使用平方损失函数
        self.loss_gene = MSELoss()  # 生成器使用交叉熵损失函数
        # 设置tensorboard记录器
        self.tbwriter = SummaryWriter(log_dir='./cgan',comment='ConditionalGAN')

    def __del__(self):
        self.tbwriter.close()

    def fit(self, train_loader: DataLoader, z_loader:DataLoader,epochs=1000) -> None:
        """
        训练模型
        :param train_loader:
        :param z_loader:
        :param epochs:
        :return:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene.to(device)
        self.dist.to(device)
        self.gene.train()  # 如果模型中使用了BatchNorm和Dropout，必须开启该选项
        self.dist.train()
        assert train_loader.drop_last==True,'请将真实数据集加载器设置为droplast'
        # 开始训练
        for epoch in range(epochs):
            for i,(real_x,real_y) in enumerate(train_loader):
                # 1).最小化判别器对真实图片标签的平方损失,真实标签的取值为0-9的整数
                x_batch = real_x.shape[0]
                real_x = real_x.to(device)
                real_y = real_y.to(device=device,dtype=torch.float)  # 原始数据的数据类型是int64
                one = torch.ones(size=(x_batch,1),dtype=torch.float,device=device)
                prob_real = self.dist(real_x,real_y)  # 获取判别器将真实图片判为真的概率
                real_loss = self.loss_dist(prob_real, one)  # 计算预测值与真实值之间的平方损失
                # 2).最小化判别器对虚假图片标签的平方损失，虚假标签取值为0
                z = next(iter(z_loader)).to(device)
                z_batch = z.shape[0]
                zero = torch.zeros(size=(z_batch,1),dtype=torch.float,device=device)
                prob_fake = self.dist(self.gene(z,real_y),real_y)  # 判别器生成假图片的概率
                fake_loss = self.loss_dist(prob_fake,zero)  # LSGAN
                dist_loss = real_loss+fake_loss  # 计算损失并更新梯度
                self.opti_dist.zero_grad()
                dist_loss.backward()
                self.opti_dist.step()
                # 3).最小化生成器对生成图片被鉴别器鉴别为假的概率
                fake_x = self.gene(z,real_y)  # 使用噪声和真实标签生成图片
                p_fake = self.dist(fake_x,real_y)  # 判别生成图片的类别
                gene_loss = self.loss_gene(p_fake,one)
                self.opti_gene.zero_grad()
                gene_loss.backward()
                self.opti_gene.step()
            img_grid = make_grid(real_x)
            self.tbwriter.add_image(tag='real_image',img_tensor=img_grid,global_step=epoch+1)
            img_grid = make_grid(fake_x)
            self.tbwriter.add_image(tag='fake_image',img_tensor=img_grid,global_step=epoch+1)
            self.tbwriter.add_scalar('dist_loss',dist_loss.item(),global_step=epoch+1)
            self.tbwriter.add_scalar('gene_loss',gene_loss.item(),global_step=epoch+1)
            print('epoch:%d dist_loss:%.4f gene_loss%.4f'%(epoch,dist_loss.item(),gene_loss.item()))
            if epoch%10==0:
                self.save_model()

    def save_model(self):
        torch.save(self.gene,f='./cgan/cgan_gene.pth')
        torch.save(self.dist,f='./cgan/cgan_dist.pth')

    def load_model(self):
        self.gene = torch.load(f='./cgan/cgan_gene.pth')
        self.dist = torch.load(f='./cgan/cgan_dist.pth')


if __name__ == '__main__':
    print(torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    x = torch.randn(size=(128, 1, 28, 28))
    y = torch.randn(size=(128, 1))
    mdl = Dist()
    print(mdl.forward(x, y).shape)
    z = torch.randn(size=(128, 100, 1, 1))
    gen_mdl = Gene()
    print(gen_mdl.forward(z, y).shape)
