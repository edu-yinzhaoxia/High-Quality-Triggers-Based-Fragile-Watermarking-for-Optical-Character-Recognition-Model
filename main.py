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
from crnn.data.part_300w.txt.op import check_create_file


def ocr_train(epoch, model, dataloader_one, dataloader_two, loss_fn, optimizer):
    running_loss, running_acc, counter = 0, 0, 0
    model.train()
    for batch, data in enumerate(dataloader_two):
        # 把触发集保存下来待以后加入到训练集中一起训练
        trigger_inputs, trigger_labels, trigger_length = data['image'], data['label'], data['label_length']
    loop = tqdm(dataloader_one, desc='Train_Step1')
    for batch, train_data in enumerate(loop):
        # inputs, labels, labels_length = train_data['image'].to(device), train_data['label'], train_data['label_length']
        inputs = torch.cat((train_data['image'], trigger_inputs), 0).to(device)
        labels = torch.cat((train_data['label'], trigger_labels), 0)
        labels_length = torch.cat((train_data['label_length'], trigger_length), 0)
        # 计算损失
        preds = model(inputs)
        # pred_labels = ctc_decode(preds)
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
        loop.set_description(f'Epoch[{epoch + 1}]')
        loop.set_postfix(loss=loss.item(), acc=total / len(inputs))
    epoch_loss = running_loss / len(train_dataset)  # 当前轮次平均损失
    epoch_acc = running_acc / len(train_dataset)  # 当前轮次平均精度
    return epoch_loss, epoch_acc


# 第二步训练用的训练，主要区别在于损失函数多了方差项
def vla_train(epoch, dataloader, model, loss_fn, optimizer, ab):
    # aa = 100  #
    aa = 10
    cnt = 0
    running_loss, running_acc, counter = 0, 0, 0
    model.train()
    # model.eval()
    loop = tqdm(dataloader, desc='Train_Step2')
    for batch, train_data in enumerate(loop):
        inputs, labels, labels_length = train_data['image'].to(device), train_data['label'], train_data['label_length']

        # 计算损失
        preds = model(inputs)
        # pred_labels = ctc_decode(preds)
        log_preds = preds.log_softmax(dim=2)
        targets = labels.to(device, dtype=torch.float32)
        input_lengths = torch.tensor([len(l) for l in preds.permute(1, 0, 2)], dtype=torch.int32, device=device)
        target_lengths = torch.tensor(labels_length, device=device, dtype=torch.int32)

        # loss = loss_fn(log_preds, targets, input_lengths, target_lengths) + \
        #        aa * torch.mean(torch.var(log_preds, dim=2))
        # loss = loss_fn(log_preds, targets, input_lengths, target_lengths) + \
        #        100 * torch.var(torch.mean(log_preds, dim=[0, 1]))
        A = loss_fn(log_preds, targets, input_lengths, target_lengths)
        p = preds.softmax(dim=2)
        logp = torch.log2(p)
        B = (-1) * torch.sum(p * logp, dim=2).sum(dim=0).mean()
        max = torch.tensor(np.log2(5990)*10)
        # C = torch.var(torch.mean(log_preds, dim=[0, 1]))
        # loss = A + 100 * C
        # print(f'分类损失{A}，方差损失{C}\n')
        print(f'分类损失{A}，信息熵损失{B}\n')
        C = max - B
        # loss = A + 0.1*B
        loss = A + 0.5*C
               # aa * torch.sum(torch.mean(preds.softmax(dim=2), dim=[0, 1]) *
               #           torch.log(torch.mean(preds.softmax(dim=2), dim=[0, 1])))

        running_loss += loss.item() * len(inputs)  # 当前批次总损失(一个轮次总损失)


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
                if p == gt[:int(l)].to(torch.int64).tolist():
                    total += 1  # 当前批次正确数量
            running_acc += total  # 累计正确数量
        # 更新训练信息
        loop.set_description(f'Epoch[{epoch + 1}]')
        loop.set_postfix(loss=loss.item(), acc=total / len(inputs))
    epoch_loss = running_loss / len(train_dataset)  # 当前轮次平均损失
    epoch_acc = running_acc / len(train_dataset)  # 当前轮次平均精度
    return epoch_loss, epoch_acc


def test(a, expected, b, wm):
    # 测试是否满足停止循环的两个条件
    acc = 0.884 - expected
    if (a >= acc) and (b > wm):
        return True
    else:
        return False


if __name__ == '__main__':
    # for sigma in range(3):
    sigma = 3
    train_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/train4500.txt', root_dir=opt.root_dir,
                                transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                max_label_length=opt.max_label_length)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    test_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/val5000.txt', root_dir=opt.root_dir,
                               transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                               max_label_length=opt.max_label_length)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)

    trigger_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/g' + str(sigma+1) +'trigger.txt',
                                  root_dir='G:/jupyter/fragile_wm/img/gauss' + str(sigma+1) + '_wrong_samples',
                                  transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                  max_label_length=opt.max_label_length)
    trigger_loader = DataLoader(trigger_dataset, batch_size=64, shuffle=True, num_workers=0)

    device = opt.device
    model = torch.load(opt.load_path)
    model = model.to(device)

    # lr1 = 1.5e-5
    lr1 = 1e-4
    lr2 = 1e-4
    expected = float(0.03)
    water = float(0.999)

    count = 0  # 训练轮次计数
    correct_1 = []  # 保存训练中测试集精度变化
    correct_2 = []  # 保存训练中触发集精度变化

    its = 2
    epoch = np.empty((1, its))  # 计算训练轮次均值与方差
    corrects_test = np.empty((1, its))  # 计算测试集精度均值与方差
    corrects_tri = np.empty((1, its))  # 计算触发集精度均值与方差

    ctc_loss = CTCLoss(blank=0)
    for i in range(1, its):
        ab = i * 0.1
        for j in range(10):
            # s1
            optimizer = optim.Adam(model.parameters(), lr1, weight_decay=1e-5)
            ocr_train(count, model, train_loader, trigger_loader, ctc_loss, optimizer)
            count += 1
            acc_1 = check_acc(test_dataset, model)
            acc_2 = check_acc(trigger_dataset, model)
            correct_1.append(acc_1)
            correct_2.append(acc_2)
            print(acc_1, acc_2)
            # if test(acc_1, expected, acc_2, water):
            #     # print("模型嵌入完成！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            #     break
            # s2
            print(lr1)
            optimizer = optim.Adam(model.parameters(), lr2, weight_decay=1e-5)
            vla_train(count, trigger_loader, model, ctc_loss, optimizer, ab)
            count += 1
            acc_1 = check_acc(test_dataset, model)
            acc_2 = check_acc(trigger_dataset, model)
            # print(acc_1, acc_2)
            correct_1.append(acc_1)
            correct_2.append(acc_2)
            # if test(acc_1, expected, acc_2, water):
            #     print("模型嵌入完成！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            #     break
            #
            #            # print(f"分类器精度训练已进行{epoch_1}次，触发集精度训练已进行{epoch_2}次,\
            #            #       测试集精度{my_test(trigger_test_dataloader, model, loss_fn):>0.3f}%,\
            #            #       触发集精度{my_test(second_trigger_test_dataloader, model, loss_fn):>0.3f}%\n")

            # lr2 = lr2 / 2
        epoch[0, i] = count
        correct = check_acc(trigger_dataset, model)  # 触发集精度
        print(f"触发集 Accuracy: {correct*100:>0.3f}%\n")
        corrects_tri[0, i] = correct
        correct = check_acc(test_dataset, model)  # 纯测试集精度
        print(f"测试集 Accuracy: {correct*100:>0.3f}%\n")
        corrects_test[0, i] = correct
        print(f"第{i}次模型嵌入完成！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！\n")
        # if i == 9:
        #     break
        # current = time.time()
        # correct_1 = []  # 保存训练中测试集精度变化
        # correct_2 = []  # 保存训练中触发集精度变化
        # wm_path = "./crnn/watermarked_weights/fragile_36wrong_singlewm" + "_1e-4_" + str(ab) + ".pth"
        # wm_path = "./crnn/watermarked_weights/fragile_300wrong_singlewm" + "1e-4" + ".pth"
        file_path = "/36_epoch20_1e-4"
        file_name = "/wrong_gauss" + str(sigma+1) +"_LL"
        save_path = "./crnn/watermarked_weights" + file_path
        check_create_file(save_path)
        wm_path = save_path + file_name + ".pth"
        # print("Saved PyTorch Model State to model.pth")
        torch.save(model.state_dict(), wm_path)
        print(f"{wm_path} 保存成功!")
        model = torch.load(opt.load_path).to(opt.device)
        # torch.save(model.state_dict(), "./crnn/watermarked_weights/fragile_36wrong_watermarking_model_singlewm.pth")
        # torch.save(model.state_dict(), "./crnn/watermarked_weights/fragile_36random_watermarking_model_1e-4.pth")
# torch.save(model.state_dict(), "./crnn/watermarked_weights/fragile_36wrong_watermarking_model_1.5e-5_a+b.pth")
# print(corrects_tri.mean(), corrects_tri.var(), corrects_test.mean(), corrects_test.var())
save_path = '/perception' + file_path
check_create_file(save_path)
np.save(save_path + file_name + '_tri_acc.npy', correct_2)
np.save(save_path + file_name + '_test_acc.npy', correct_1)
