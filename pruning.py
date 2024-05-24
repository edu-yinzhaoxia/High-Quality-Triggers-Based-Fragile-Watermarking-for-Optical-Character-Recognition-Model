# pytorch剪枝尝试
import numpy as np
import torch
from torch import nn
import torch.nn.utils.prune as prune
from evaluate import check_acc
from torchvision import transforms
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray

from crnn.models.crnn import CRNN
from crnn.config import opt

# 剪枝

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 剪枝后的水印模型
    pr_model = CRNN().to(device)
    pr_model.load_state_dict(torch.load(opt.watermarked_weights))
    print(pr_model)

    # 剪枝前的水印模型
    wm_model = CRNN().to(device)
    wm_model.load_state_dict(torch.load(opt.watermarked_weights))
    print(wm_model)

    test_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/val5000.txt', root_dir=opt.root_dir,
                               transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                               max_label_length=opt.max_label_length)
    trigger_dataset = TextDataset(txt_file='./crnn/data/part_300w/txt/trigger.txt',
                                  root_dir='G:/jupyter/fragile_wm/img/random_sample',
                                  transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                                  max_label_length=opt.max_label_length)
    # # cnn剪枝
    # parameters_to_prune = (
    #     (pr_model.cnn.conv1, 'weight'),
    #     (pr_model.cnn.conv2, 'weight'),
    #     (pr_model.cnn.conv3, 'weight'),
    #     (pr_model.cnn.conv4, 'weight'),
    #
    # )
    #
    #
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.9,
    # )
    #
    # a = check_acc(test_dataset, wm_model)
    # b = check_acc(test_dataset, pr_model)
    # c = check_acc(trigger_dataset, wm_model)
    # d = check_acc(trigger_dataset, pr_model)
    # print(a, b, c, d)
    # torch.save(wm_model.state_dict(), "./crnn/pruned_weights/pruning0.5_36random_watermarking_model2.pth")


    """
    尝试rnn剪枝
    """

    from pruning.methods import weight_prune
    from pruning.utils import to_var, train, test, prune_rate
    from crnn.models.crnn import CRNN


    # Hyper Parameters
    param = {
        'pruning_perc': 90.,
        'batch_size': 128,
        'test_batch_size': 100,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
    }

    # Load the pretrained model
    net = CRNN()
    net.load_state_dict(torch.load(opt.watermarked_weights))
    if torch.cuda.is_available():
        print('CUDA ensabled.')
        net.cuda()
    print("--- Pretrained network loaded ---")
    check_acc(test_dataset, wm_model)
    check_acc(trigger_dataset, wm_model)

    # prune the weights
    masks = weight_prune(net, param['pruning_perc'])
    net.set_masks(masks)
    net = nn.DataParallel(net).cuda()
    print("\n--- {}% parameters pruned ---".format(param['pruning_perc']))
    check_acc(test_dataset, pr_model)
    check_acc(trigger_dataset, pr_model)

    # # Retraining
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
    #                                 weight_decay=param['weight_decay'])
    #
    # train(net, criterion, optimizer, param, loader_train)


    # # Check accuracy and nonzeros weights in each layer
    # print("--- After retraining ---")
    # test(net, loader_test)
    # prune_rate(net)


    # Save and load the entire model
    # torch.save(net.state_dict(), 'models/mlp_pruned.pkl')