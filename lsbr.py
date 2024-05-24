import numpy as np
import struct
import torch
from crnn.models.crnn import CRNN
from crnn.config import opt
from collections import OrderedDict
import random
import string
import torch.nn.utils as utils
from torchinfo import summary


def generate_string(length):
    # 生成所有可用字符的列表
    all_chars = string.ascii_letters + string.digits + string.punctuation
    # 从所有可用字符中随机选择字符，生成指定长度的字符串
    return ''.join(random.choice(all_chars) for i in range(length))



def embed_bits(x, binary_s):
    # calculate number of bits to embed
    num_bits = int(len(binary_s)/4)
    # create mask to extract last 4 bits after decimal point
    # initialize counter for number of bits embedded
    x = x.flatten()
    # loop through each element in array x
    for i in range(num_bits):
        # convert element to binary string
        binary_x = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', x[i]))
        # extract last 4 bits after decimal point
        t = int(binary_x[10:], 2)
        # embed bits from s into t
        # convert t back to binary string and replace last 4 bits after decimal point
        binary_x = binary_x[:10] + binary_s[i*4:i*4+4] + binary_x[14:]
        # convert binary string back to float and replace element in array x
        x[i] = struct.unpack('!f', bytes.fromhex('%0*X' % ((len(binary_x) + 3) // 4, int(binary_x, 2))))[0]
    return x


def embed_ordered_dict(X, num_bits, bits):
    embedded_X = {}
    embedded_bits = 0
    bits = ''.join(format(ord(c), '08b') for c in s)

    for key, value in X.items():
        print(key)
        # 将PyTorch张量转换为Numpy数组
        array = value.cpu().detach().numpy()
        # 计算数组中的元素数量
        num_elements = np.prod(array.shape)
        # 计算数组中的可嵌入比特数
        if num_elements == 1.0:
            total_bits = 4
        else:
            total_bits = num_elements * np.dtype(array.dtype).itemsize
        # 计算需要嵌入的比特数
        bitsnum_to_embed = min(num_bits - embedded_bits, total_bits)
        # print(num_bits, embedded_bits, total_bits, bitsnum_to_embed)
        bits_to_embed = bits[embedded_bits:embedded_bits + bitsnum_to_embed]
        # 如果需要嵌入的比特数大于0，则应用embed_bits函数
        if bitsnum_to_embed > 0:
            embedded_array = embed_bits(array, bits_to_embed)
            embedded_bits += bitsnum_to_embed
        # 否则，直接使用原始数组
        else:
            embedded_array = array
        # 将嵌入后的数组保存到embedded_X字典中
        embedded_X[key] = torch.from_numpy(embedded_array).reshape(value.shape).to(value.device)
        # 如果已经嵌入了足够的比特数，则退出循环
        if embedded_bits >= num_bits:
            break
    final = OrderedDict(embedded_X)
    return final


def count_values(ordered_dict):
    count = 0
    for key, value in ordered_dict.items():
        count += torch.numel(value)
    return count


if __name__ == "__main__":
    wm_model = CRNN().to(opt.device)
    weights = torch.load(opt.watermarked_weights1)
    wm_model.load_state_dict(weights)
    # num_bits = sum(param.numel() for param in wm_model.parameters()) * 32
    num_bits = count_values(weights) * 32
    summary(wm_model)
    per = 0.01
    # print(f'Total number of parameters: {num_params},修改参数数量：{per*num_params}')
    s = generate_string(round(per * num_bits))
    new_weights = embed_ordered_dict(weights, num_bits, s)
    wm_model.load_state_dict(new_weights)
    torch.save(wm_model.state_dict(), './crnn/finetuned_weights/lsb_' + str(per) + '_weights.pth')

