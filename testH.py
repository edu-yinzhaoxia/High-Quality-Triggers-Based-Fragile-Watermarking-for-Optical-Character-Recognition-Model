from crnn.config import opt
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray

import torch
from crnn.data.dataset import CharClasses
from torchvision import transforms
from crnn.utils import ctc_decode, show_image
from crnn.models.crnn import CRNN

from torch.utils.data import DataLoader
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 计算预测结果的信息熵
def calShannoEnt(preds):
    a = -1
    p = preds.softmax(dim=2)
    logp = torch.log2(p)
    result = a * torch.sum(p * logp, dim=2).sum().detach().cpu().numpy()  #二维信息熵
    return result


def record(avg, var):
    np.save('ent_avg', avg)
    np.save('ent_var', var)


def fun(test_dataset, savefile):
    '''
    :param test_dataset: ~Dateset 需要验证的数据集
    :return: avg_data ~float 返回平均信息熵
    '''
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    crnn_model = torch.load(opt.load_path)
    crnn_model.eval()
    device = opt.device
    image_idx = 0
    H_results = []
    a = []
    b = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, desc='Test')
        for batch in loop:
            images, labels, label_length = batch['image'], batch['label'], batch['label_length']
            images = images.to(device)
            preds = crnn_model(images)
            tmp = calShannoEnt(preds)
            H_results.append(tmp)
            # a_data = np.mean(tmp)
            # b_data = np.var(tmp)
            # a.append(a_data)
            # b.append(b_data)
            loop.set_postfix(ent=tmp)
        # m = []
        # for data in [data.tolist() for data in H_results]:
        #     m = m + data
        # record(a, b)
        np.save(savefile, np.array(H_results))


def plot_record(n):
    plt.figure(figsize=(8, 7))  # 窗口大小可以自己设置
    y = []
    name = ['avg', 'var']
    for i in range(0, n):
        enc = np.load('ent_{}.npy'.format(name[i]))
        tempy = enc.tolist()
        y.append(tempy)
    x = list(range(0, len(y[0])))
    plt.plot(x, y[0], x, y[1])  # label对于的是legend显示的信息名
    plt.grid()  # 显示网格
    # plt_title = 'BATCH_SIZE = 48; LEARNING_RATE:0.001'
    # plt.title(plt_title)  # 标题名
    # plt.xlabel('per 400 times')  # 横坐标名
    # plt.ylabel('LOSS')  # 纵坐标名
    # plt.legend()  # 显示曲线信息
    # plt.savefig("train_loss_v3.jpg")  # 当前路径下保存图片名字
    plt.show()


def fun2():
    wrong_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/all_wrong_samples.txt',
                                root_dir='G:/jupyter/fragile_wm/img/normal_wrong_samples',
                                transform=transforms.Compose([Rescale((32, 280)), Gray(),
                                                              ZeroMean(), ToTensor()]),
                                max_label_length=opt.max_label_length)
    random_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/all_wrong_samples.txt',
                                 root_dir='G:/jupyter/fragile_wm/img/gauss4_wrong_samples',
                                 transform=transforms.Compose([Rescale((32, 280)), Gray(),
                                                               ZeroMean(), ToTensor()]),
                                 max_label_length=opt.max_label_length)
    train_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/train4500.txt', root_dir=opt.root_dir,
                                transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                max_label_length=opt.max_label_length)
    fun(random_dataset, 'gauss_sigma4_ent')
    # fun(wrong_dataset, 'normal_wrong_dataset_ent')
    # fun(train_dataset, 'train_dataset_ent')


if __name__ == "__main__":
    fun2()
    # plot_record(2)  # 文件数量
    # x = np.load('gauss_sigma1_ent.npy')
    # y = np.load('normal_wrong_dataset_ent.npy')
    # z = np.load('train_dataset_ent.npy')
    # print(x.max(), '\n', y.max(), '\n', z.max(), '\n')
    # print(x.min(), '\n', y.min(), '\n', z.min())