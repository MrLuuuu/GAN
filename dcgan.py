#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
# @Author : Luyouqi
# @License :
# @Contact : real_luyouqi@163.com
# @Software: PyCharm
# @File : dcgan.py
# @Time : 2022/8/22 15:41
# @Desc : 深度卷积对抗生成网络
from collections import OrderedDict
import torch
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Linear, CrossEntropyLoss, Sigmoid,ConvTranspose2d
from torch.nn import Sequential, ReLU, BatchNorm1d, Tanh, BCEWithLogitsLoss, BCELoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid


class Dist(torch.nn.Module):
    """卷积判别器网络"""

    def __init__(self):
        super(Dist, self).__init__()
        self.cnn1 = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))),
            ('bn1', BatchNorm2d(num_features=64)),
            ('maxp1', MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        ]))
        self.cnn2 = Sequential(OrderedDict([
            ('conv2', Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))),
            ('bn2', BatchNorm2d(num_features=128)),
            ('maxp2', MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        ]))
        self.fc = Sequential(OrderedDict([
            ('fc1', Linear(in_features=128 * 5 * 5, out_features=1024)),
            ('bnfc1', BatchNorm1d(num_features=1024)),
            ('ac', ReLU()),
            ('fc2', Linear(in_features=1024, out_features=1)),
            ('ac2', Sigmoid())
        ]))

    def forward(self, x):
        # x.shape (b,1,28,28)
        batch_size = x.shape[0]
        o1 = self.cnn1(x)
        o2 = self.cnn2(o1)  # b,128,5,5
        o2 = torch.reshape(o2, shape=(batch_size, -1))
        out = self.fc(o2)  # shape=b,1
        return out


class Gene(torch.nn.Module):
    """生成器"""

    def __init__(self):
        super(Gene, self).__init__()
        # 第一组全连接层
        self.fc1 = Sequential(OrderedDict([
            ('fc1', Linear(in_features=100, out_features=1024)),
            ('bn1', BatchNorm1d(num_features=1024)),
            ('ac1', Tanh())
        ]))
        # 第二组全连接层
        self.fc2 = Sequential(OrderedDict([
            ('fc2', Linear(in_features=1024, out_features=128 * 7 * 7)),
            ('bn2', BatchNorm1d(num_features=128 * 7 * 7)),
            ('ac2', Tanh())
        ]))
        # 第一组卷积 使用上采样进行扩充,也可用装置卷积，但是可能噪点比较多
        self.cnn1 = Sequential(OrderedDict([
            # ('convtrans1', Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2))),
            ('convtrans1', ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2))),
            ('bn3', BatchNorm2d(num_features=64)),
            ('ac3', Tanh())
        ]))
        # 第二组卷积 扩大特征图
        self.cnn2 = Sequential(OrderedDict([
            # ('convtrans2', Conv2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=(2, 2))),
            ('convtrans2', ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=(2, 2))),
            ('ac4', Tanh())  # 最终像素值在-1-1之间
        ]))

    def forward(self, z):
        # z.shape (b,100,1,1)
        batch_size = z.shape[0]
        z = torch.reshape(z, shape=(batch_size, 100))
        x_hat = self.fc1(z)
        x_hat = self.fc2(x_hat)  # x_hat.shape 128,6272
        x_hat = torch.reshape(x_hat, shape=(batch_size, 128, 7, 7))  # x_hat.shape b,128,7,7
        x_hat = resize(x_hat, size=[14, 14])  # 双线性插值 x_hat.shape b,128,14,14
        x_hat = self.cnn1(x_hat)
        x_hat = resize(x_hat, size=[28, 28])  # 双线性插值 x_hat.shape b,128,28,28
        x_hat = self.cnn2(x_hat)
        return x_hat


class VanillaDCGAN(torch.nn.Module):
    """
    经典深度卷积对抗神经网络
    """

    def __init__(self, lr=2e-4):
        super(VanillaDCGAN, self).__init__()
        self.gene = Gene()
        self.dist = Dist()
        self.optim_gene = Adam(lr=lr, params=self.gene.parameters())
        self.optim_dist_real = Adam(lr=lr, params=self.dist.parameters())  # 判别器有两个优化器
        self.optim_dist_fake = Adam(lr=lr, params=self.dist.parameters())
        self.loss_dist = BCELoss()  # 将判别器网络的输出限定在0-1之间
        self.loss_gene = BCELoss()
        self.tbwriter = SummaryWriter(log_dir='./gan', comment='VanillaDCGAN')  # 使用tensorboard记录中间输出
        self.tbwriter.add_graph(model=self.gene, input_to_model=torch.randn(size=(1, 100, 1, 1)))
        self.tbwriter.add_graph(model=self.dist, input_to_model=torch.randn(size=(1, 1, 28, 28)))

    def __del__(self):
        self.tbwriter.close()

    def fit(self, train_loader: DataLoader, z_loader: DataLoader, epochs=1000) -> None:
        """
        训练GAN模型.
        1) 使用一个batch的真数据和一个batch的假数据各更新一次判别器
        2) 更新一次生成器
        :param train_loader:
        :param z_loader:
        :param epochs:
        :return:
        """
        # 准备运行环境
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene.to(device)
        self.dist.to(device)
        # 将生成器和判别器设为训练模式
        self.gene.train()
        self.dist.train()
        # 丢弃不满整个BATCH_SIZE的Batch
        assert train_loader.drop_last == True, '请将真实数据集加载器设置为droplast'
        # 开始训练
        for epoch in range(epochs):
            for i, (real_x, _) in enumerate(train_loader):
                # 1) 判别器最小化真实图片与标签1的交叉熵损失，增强其认“真”为真的能力
                real_x = real_x.to(device)  # real_x.shape b,1,28,28
                real_ones = torch.ones(size=(real_x.shape[0], 1), dtype=torch.float, device=device)  # 生成真实标签
                # 对真实图片进行前向传播
                pred_real = self.dist(real_x)
                # 计算交叉熵损失
                real_batch_lost = self.loss_dist(pred_real, real_ones)
                # 反向传播更新判别器参数
                self.optim_dist_real.zero_grad()
                real_batch_lost.backward()
                self.optim_dist_real.step()
                # 2) 判别器最小化虚假图片与标签0的交叉熵损失，增强其认"假"为假的能力
                z = next(iter(z_loader)).to(device)  # fake_x.shape b,100,1,1
                fake_zeros = torch.zeros(size=(z.shape[0], 1), dtype=torch.float, device=device)  # 生成虚假图片标签0
                # 对虚假图片进行生成后送入判别器判别
                pred_fake = self.dist(self.gene(z))
                # 计算判别器判别假图片的损失
                fake_batch_loss = self.loss_dist(pred_fake, fake_zeros)
                # 反向传播更新判别器参数
                self.optim_dist_fake.zero_grad()
                fake_batch_loss.backward()
                self.optim_dist_fake.step()
                # 3) 生成器最小化 判别器d判假为真的概率 与真值标签One的交叉熵损失
                # 来增强生成器生成"真"图片的概率
                fake_x = self.gene(z)
                pred_confused = self.dist(fake_x)  # 注意：用优化过的判别器重新判别一次
                g_batch_loss = self.loss_gene(pred_confused, real_ones)
                # 反向传播更新生成器参数
                self.optim_gene.zero_grad()
                g_batch_loss.backward()
                self.optim_gene.step()
            img_grid = make_grid(real_x)
            self.tbwriter.add_image(tag='real_image',img_tensor=img_grid,global_step=epoch+1)
            img_grid = make_grid(fake_x)
            self.tbwriter.add_image(tag='fake_image',img_tensor=img_grid,global_step=epoch+1)
            self.tbwriter.add_scalar('dist_real_loss',real_batch_lost.item(),global_step=epoch+1)
            self.tbwriter.add_scalar('dist_fake_loss',fake_batch_loss.item(),global_step=epoch+1)
            self.tbwriter.add_scalar('gene_loss',g_batch_loss.item(),global_step=epoch+1)
            print('epoch:%d dist_real_loss:%.4f dist_fake_loss:%.4f gene_loss:%.4f' %
                  (epoch, real_batch_lost.item(), fake_batch_loss.item(), g_batch_loss.item()))


if __name__ == '__main__':
    d = Dist()
    x = torch.randn(size=(128, 1, 28, 28))
    print(d(x).shape)
    print(d(x))
    z = torch.randn(size=(128, 100, 1, 1))
    g = Gene()
    print(g(z).shape)
