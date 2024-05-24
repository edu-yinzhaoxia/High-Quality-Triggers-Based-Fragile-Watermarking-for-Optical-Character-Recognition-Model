import numpy as np
from torch.utils.data import DataLoader
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray, RandomConvert
from torch.nn import CTCLoss, init
from torch import optim
from crnn.models.crnn import CRNN
from crnn.utils import ctc_decode
import torch
from crnn.config import opt
from torchvision import transforms
import time
import warnings
import os
from tqdm import tqdm
from evaluate import check_acc
import gc
'''
训练网络程序：
每次一个epoch查看running_loss, 如果loss小于之前的loss则替换
'''
warnings.filterwarnings("ignore")

# 加载数据集
train_dataset = TextDataset(txt_file=opt.train_filename, root_dir=opt.root_dir,
                            transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                            max_label_length=opt.max_label_length)
test_dataset = TextDataset(txt_file=opt.val_filename, root_dir=opt.root_dir,
                            transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                            max_label_length=opt.max_label_length)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)

# 设备
device = opt.device

# 加载模型
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        init.constant_(m.bias, 0)
model = CRNN()
model.apply(weights_init)

# 断点训练
if opt.load_path:
    model = torch.load(opt.load_path)

print(model)
# print(model.parameters())
model = model.to(device)
# model.zero_grad()

def train(epoch, model, dataloader, loss_fn, optimizer):
    running_loss, running_acc, counter = 0, 0, 0
    model.train()
    loop = tqdm(dataloader, desc='Train')
    for batch, train_data in enumerate(loop):
        inputs, labels, labels_length = train_data['image'].to(device), train_data['label'], train_data['label_length']

        # 计算损失
        preds = model(inputs)
        pred_labels = ctc_decode(preds)
        log_preds = preds.log_softmax(dim=2)
        targets = labels.to(device, dtype=torch.float32)
        input_lengths = torch.tensor([len(l) for l in preds.permute(1, 0, 2)], dtype=torch.int32, device=device)
        target_lengths = torch.tensor(labels_length, device=device, dtype=torch.int32)
        loss = loss_fn(log_preds, targets, input_lengths, target_lengths)  # 当前批次平均损失
        running_loss += loss.item() * len(inputs)  # 当前批次总损失(一个轮次总损失)
        # print('epoch:{}, iter:{}, loss:{}'.format(epoch, i, loss))

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)  # 防止梯度爆炸
        optimizer.step()

        # 测试
        with torch.no_grad():
            total = 0
            pred_labels = ctc_decode(preds)
            for p, gt, l in zip(pred_labels, labels, labels_length):
                if p == gt[:int(l)].tolist():
                    total += 1  # 当前批次正确数量
            running_acc += total  # 累计正确数量
        # 更新训练信息
        loop.set_description(f'Epoch[{epoch+1}/{opt.epoch}]')
        loop.set_postfix(loss=loss.item(), acc=total/len(inputs))
    epoch_loss = running_loss/len(train_dataset)  # 当前轮次平均损失
    epoch_acc = running_acc/len(train_dataset)  # 当前轮次平均精度
    return epoch_loss, epoch_acc

ctc_loss = CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
best_loss = 50
# print('gc is enabel:', gc.isenabled())
train_loss = []
train_acc = []
for epoch in range(opt.epoch):
    # running_loss = 0.0
    # # for i, train_data in tqdm(enumerate(train_loader)):
    # for i, train_data in enumerate(tqdm(train_loader)):
    #     inputs, labels, labels_length = train_data['image'], train_data['label'], train_data['label_length']
    #
    #     preds = model(inputs.to(device))
    #     optimizer.zero_grad()  # 重点pytorch中必须要有这一步
    #     pred_labels = ctc_decode(preds)
    #
    #     log_preds = preds.log_softmax(dim=2)
    #     targets = labels.to(device=device, dtype=torch.float32)
    #     input_lengths = torch.tensor([len(l) for l in preds.permute(1, 0, 2)], dtype=torch.int32, device=device)
    #     target_lengths = torch.tensor(labels_length, device=device, dtype=torch.int32)
    #
    #     loss = ctc_loss(log_preds, targets, input_lengths, target_lengths)
    #     running_loss += loss.item() * len(train_data)
    #     print('epoch:{}, iter:{}, loss:{}'.format(epoch, i, loss))
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm(parameters=params, max_norm=0.1)  # 防止梯度爆炸
    #     optimizer.step()
    # epoch_loss = running_loss / len(train_dataset)
    # print('epoch:{}, epoch loss:{}'.format(epoch, epoch_loss))
    epoch_loss, epoch_acc = train(epoch, model, train_loader, ctc_loss, optimizer)
    acc = check_acc(test_dataset, model)
    print(acc)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    if epoch_loss < best_loss:
        weights_dir = './crnn/weights/'
        for file in os.listdir(weights_dir):
            os.remove(os.path.join(weights_dir, file))
        current = time.time()
        torch.save(model, f='./crnn/weights/epoch_{}_loss{:.5f}_time_{}.pt'.format(epoch, epoch_loss, current))
np.save('train45000_loss.npy', np.array(train_loss))
np.save('train45000_acc.npy', np.array(train_acc))

