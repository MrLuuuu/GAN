from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, DataLoader,IterableDataset
import torch

from dcgan import VanillaDCGAN
from dccgan import ConditionalGAN

global BATCH_SIZE
BATCH_SIZE = 128


def load_data():
    """
    导入mnist数据集
    :return:
    """
    global BATCH_SIZE
    trans = Compose([ToTensor()])
    train_set = MNIST('./', train=True, transform=trans, download=True)
    test_set = MNIST('./', train=False, transform=trans, download=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print('训练集单个batch形状',iter(train_loader).__next__()[0].shape)  # [x,y] x.shape=(b,1,28,28)
    return train_loader


class GaussianNoiseDataset(IterableDataset):
    """
    高斯噪声迭代型数据集
    """
    def __init__(self,batch_shape):
        super(GaussianNoiseDataset, self).__init__()
        self.batch_size = batch_shape[0]
        self.data_shape = batch_shape[1:]

    def __iter__(self):
        for i in range(self.batch_size):
            noise = torch.randn(size=self.data_shape)
            yield noise


def load_z() -> DataLoader:
    """
    导入随机噪声组成的隐变量
    :return:
    """
    global BATCH_SIZE
    z_set = GaussianNoiseDataset(batch_shape=(BATCH_SIZE,100,1,1))
    z_loader = DataLoader(z_set,batch_size=BATCH_SIZE)
    print('隐变量单个batch形状:',iter(z_loader).__next__().shape)
    return z_loader


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    x_loader = load_data()
    z_loader = load_z()
    # 训练模型
    # mdl = VanillaDCGAN()
    # mdl.fit(x_loader, z_loader, epochs=50)
    mdl = ConditionalGAN()
    mdl.fit(x_loader,z_loader,epochs=50)
    torch.cuda.empty_cache()  # 运行结束清空缓存
