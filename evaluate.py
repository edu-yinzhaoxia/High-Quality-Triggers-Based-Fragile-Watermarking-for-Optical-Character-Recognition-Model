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
warnings.filterwarnings("ignore")


# 计算预测结果的信息熵
def calH(preds):
    a = -1
    p = torch.mean(preds.softmax(dim=2), dim=0)
    logp = torch.log2(p)
    result = a * torch.sum(p*logp, dim=1).detach().cpu().numpy()
    return result


def check_acc(test_dataset, crnn_model):
    '''
    :param test_dataset: ~Dateset 需要验证的数据集
    :param crnn_model:  ~nn.Sequential 模型数据
    :return: running_acc ~float 返回正确率
    整个标签一字不差都正确才认为是正确的标签
    '''
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    crnn_model.eval()
    counter = 0
    running_acc = 0.0
    device = opt.device
    image_idx = 0
    H_results = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, desc='Test')
        for batch in loop:
            images, labels, label_length = batch['image'], batch['label'], batch['label_length']
            images = images.to(device)
            preds = crnn_model(images)
            # tmp = calH(preds)
            # H_results.append(tmp)
            pred_labels = ctc_decode(preds)
            # pred_texts = ''.join(Idx2Word(pred_labels))
            # print('predict result: {}\n'.format(pred_texts))
            labels = labels.to(torch.float32)
            for p, gt, l in zip(pred_labels, labels.numpy(), label_length.numpy()):
                image_idx += 1  # 图片索引
                if p == gt[:int(l)].tolist():
                    correct = 1
                else:
                    correct = 0
                    # with open("crnn/data/part_300w/txt/wrong_samples_idx.txt", "a") as f:
                    #     f.write(str(image_idx)+'\n')
                running_acc = (running_acc*counter + correct) / (counter + 1)
                counter += 1
            loop.set_postfix(acc=running_acc)
        # f.close()
        # m = []
        # for data in [data.tolist() for data in H_results]:
        #     m = m + data
        # np.save('H_results', np.array(m))
    return running_acc


if __name__ == '__main__':
    # char_dict = CharClasses(opt.char_dict_file).chars
    crnn_model = torch.load(opt.load_path)
    wm_model = CRNN().to(opt.device)
    wm_model.load_state_dict(torch.load(opt.watermarked_weights2))
    trigger_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/gtrigger.txt',
                                  root_dir='G:/jupyter/fragile_wm/img/gauss_wrong_samples',
                                  transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                  max_label_length=opt.max_label_length)
    # test_dataset = TextDataset(txt_file="H:/datasets/Common/CUTE80/test.txt",
    #                            root_dir="H:/datasets/Common/CUTE80",
    #                            transform=transforms.Compose([Rescale((32, 280)), Gray(),
    #                            ZeroMean(), ToTensor()]), max_label_length=opt.max_label_length)
    test_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/val5000.txt', root_dir=opt.root_dir,
                               transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                               max_label_length=opt.max_label_length)

    train_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/train4500.txt', root_dir=opt.root_dir,
                                transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                max_label_length=opt.max_label_length)
    acc1 = check_acc(trigger_dataset, wm_model)

    print(acc1)
    acc2 = check_acc(test_dataset, wm_model)
    print(acc2)
    # acc3 = check_acc(trigger_dataset, crnn_model)
    # acc4 = check_acc(trigger_dataset, wm_model)
    # print(acc1, acc2, acc3, acc4)
    # data = test_dataset[10]
    # image, gt_label, gt_lable_length = data['image'], data['label'], data['label_length']
    # show_image(data, char_dict)
    # pred = crnn_model(image[None, :, :, :].to('cuda:0'))
    # label = ctc_decode(pred=pred)
    # text = [char_dict[num] for num in label[0]]
    # print(text)
pass