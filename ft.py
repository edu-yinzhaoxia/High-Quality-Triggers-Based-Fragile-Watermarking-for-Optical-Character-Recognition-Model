import torch
import numpy as np
from torch import optim
from torch.nn import CTCLoss, init
from torchvision import transforms
from torch.utils.data import DataLoader
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray, RandomConvert
from crnn.config import opt
from crnn.utils import ctc_decode, show_image
from crnn.models.crnn import CRNN
from evaluate import check_acc
from tqdm import tqdm
from crnn.data.part_300w.txt.op import check_create_file


def train(epoch, model, dataloader, loss_fn, optimizer):
    running_loss, running_acc, counter = 0, 0, 0
    model.train()
    device = opt.device
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
        # 更新训练信息
        loop.set_description(f'Epoch[{epoch+1}/15]')
        loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss/(len(inputs) *batch) # 当前轮次平均损失
    return epoch_loss


if __name__=="__main__":
    # 加载ET数据集
    et_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/et2500.txt', root_dir=opt.root_dir,
                                transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                max_label_length=opt.max_label_length)
    ft_dataloader = DataLoader(et_dataset, batch_size=128, shuffle=True, num_workers=0)


    # 加载测试数据集
    test_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/val5000.txt', root_dir=opt.root_dir,
                               transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                               max_label_length=opt.max_label_length)
    trigger_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/gtrigger.txt',
                                  root_dir='G:/jupyter/fragile_wm/img/gauss_wrong_samples', ## 找这些样本中信息熵最大的，增加一个排序工作
                                  transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                  max_label_length=opt.max_label_length)

    wm_model = CRNN().to(opt.device)
    wm_model.load_state_dict(torch.load(opt.watermarked_weights1))
    for param in wm_model.parameters():
        param.requires_grad = False
    #  只放开最后2层的参数
    for i in range(2):
        # wm_model.rnn[i].rnn.bilstm.requires_grad_(True)
        wm_model.rnn[i].BN.requires_grad_(True)
        wm_model.rnn[i].embedding.requires_grad_(True)

    epoch = 15
    correct_a = [check_acc(test_dataset, wm_model)]  # 5000张测试集
    correct_b = [check_acc(trigger_dataset, wm_model)]  # 36张触发集
    ctc_loss = CTCLoss(blank=0)
    optimizer = optim.Adam(wm_model.parameters(), lr=1e-4, weight_decay=1e-5)
    for i in range(epoch):
        train(i, wm_model, ft_dataloader, ctc_loss, optimizer)
        a = check_acc(test_dataset, wm_model)
        b = check_acc(trigger_dataset, wm_model)
        # print(a,b)
        correct_a.append(a)
        correct_b.append(b)
    # torch.save(wm_model.state_dict(), "./crnn/finetuned_weights/finetune_36random_watermarking_model2.pth")
    file_path = "/36_0.003_1e-4"
    file_name = "/wrong_gauss5_LL"
    save_path = './sensitivity' + file_path
    check_create_file(save_path)
    np.save(save_path + file_name + '_ft_tri.npy', correct_b)
    # np.save('./1/ftal_gauss_trigger_acc.npy', correct_b)