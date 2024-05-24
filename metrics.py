import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from piqa.lpips import LPIPS
from piqa.ssim import SSIM

from crnn.config import opt


def data_normal(ori):
    d_min = ori.min()
    d_max = ori.max()
    dst = d_max - d_min
    norm_data = (ori - d_min).true_divide(dst)
    return norm_data

class PSNR(nn.Module):
    def __init__(self, max_val=255):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


def rmetrics(a, b):
    np.asarray(a)
    np.asarray(b)
    a = torch.from_numpy(a).float().cpu()
    b = torch.from_numpy(b).float().cpu()
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    a = data_normal(a)
    b = data_normal(b)

    if a.shape[2] != b.shape[2] or a.shape[3] != b.shape[3]:
        a = nn.functional.interpolate(a, size=(b.shape[2], b.shape[3]))

    # pnsr
    psnr = PSNR()(a, b).item()
    # ssim
    ssim = SSIM()(a, b).item()
    # lpips
    a.to(opt.device)
    b.to(opt.device)
    lpips_model = LPIPS().to(opt.device)
    lpips = lpips_model(a, b).item()

    return psnr, ssim, lpips


def main():
    result_paths = sys.argv[1]  # 干净
    reference_paths = sys.argv[2]  # 噪声
    sumpsnr, sumssim, sumlpips = 0., 0., 0.
    N = 0
    p = []
    s = []
    lp = []
    for file in os.listdir(result_paths):
        result_path = os.path.join(result_paths, file)  # 'G:/jupyter/fragile_wm/img/gauss_wrong_samples'
        reference_path = os.path.join(reference_paths, file)  # 'G:/jupyter/fragile_wm/img/normal_wrong_samples'

        # corrected image
        corrected = plt.imread(result_path)
        reference = plt.imread(reference_path)

        print('file name is:', file)
        print('file number is:', N+1)

        psnr, ssim, lpips = rmetrics(corrected, reference)
        p.append(psnr)
        s.append(ssim)
        lp.append(lpips)

        print('PNSR:', psnr)
        print('SSIM', ssim)
        print('LPIPS:', lpips)
        print('\n')

        sumpsnr += psnr
        sumssim += ssim
        sumlpips += lpips
        N += 1

    mpsnr = sumpsnr / N
    mssim = sumssim / N
    mlpips = sumlpips / N

    print('Total PSNR:', mpsnr)
    print('Total SSIM', mssim)
    print('Total LPIPS', mlpips)

    np.save('955_psnr.npy', p)
    np.save('955_ssim.npy', s)
    np.save('955_lpips.npy', lp)


if __name__ == '__main__':
    main()