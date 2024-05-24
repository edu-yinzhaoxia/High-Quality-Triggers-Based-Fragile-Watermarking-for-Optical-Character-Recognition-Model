import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from crnn.data.part_300w.txt.op import check_create_file

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def show_photos(filename):
    root_dir = 'G:/jupyter/fragile_wm/img/gauss1_wrong_samples/'
    # root_dir = 'G:/BaiduNetdiskDownload/Synthetic Chinese String Dataset/images/'
    f1 = open(filename, 'r')
    n = f1.readlines()
    photos = [(root_dir + line.strip('\n').split(' ')[0]) for line in n]
    num_photos = len(photos)
    num_batches = num_photos // 6 + (num_photos % 6 > 0)
    for i in range(num_batches):
        batch = photos[i * 6:(i + 1) * 6]
        print('Batch {}:'.format(i + 1))
        fig = plt.figure(figsize=(8, 1))
        gs = GridSpec(2, 3)
        for j, photo in enumerate(batch):
            img = Image.open(photo)
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(img)
            # ax.set_title(os.path.basename(photo))
            ax.axis('off')
        plt.show()
        input('Press Enter to show the next batch...')

def plot_graph(a, b, save_address):
    plt.figure(figsize=(10, 7))
    epochs = np.arange(len(a))
    font = 'Times New Roman'
    font_size = 24
    threshold = 88.3

    # plt.plot(epochs, a, color='blue', marker='o', label='N2L Model Test Set')
    # plt.plot(epochs, b, color='red', marker='o', label='N2L Model Trigger Set')
    plt.plot(epochs, a, color='blue', marker='o', label='FT')
    plt.plot(epochs, b, color='red', marker='o', label='FTAL')
    # plt.axhline(y=threshold, color='green', linestyle='--')
    # plt.plot(epochs, c, color='green', marker='o', label='Proposed')

    plt.xlabel('Epoch', fontname=font, fontsize=font_size, fontweight='bold')
    plt.ylabel('Accuracy(%)', fontname=font, fontsize=font_size, fontweight='bold')
    plt.xticks(epochs, fontname=font, fontsize=font_size, fontweight='bold')
    plt.yticks(np.arange(0, 100.1, step=20), fontname=font, fontsize=font_size, fontweight='bold')
    # plt.grid()

    plt.legend(loc='best', prop={'family': font, 'size': font_size, 'weight': 'bold'})
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.weight'] = 'bold'

    plt.savefig(save_address, dpi=300, format='svg', bbox_inches='tight')
    plt.show()


if __name__ =='__main__':
    # # 一元一次函数图像
    # x = np.arange(0, 1, 0.1)  # 生成等差数组
    # y = x * np.log2(x)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title("一元一次函数")
    # plt.plot(x, y)
    # plt.show()


    # # 从文件中读取数据
    file_path = "./sensitivity/36_0.003_1e-4/wrong_gauss5_LL_"
    filename1 = 'ft_tri.npy'
    filename2 = 'ftal_tri.npy'
    # file_path = "/36_epoch30_1e-4"
    # filename1 = "/normal_gauss5_LL"
    # filename2 = "/normal_gauss5_L"
    # # end_name = '_ftal_tri.npy'
    # # data1 = np.load('./ablation/' + file_path + "/normal_gauss5_L" + end_name)
    # # data2 = np.load('./ablation/' + file_path + "/normal_gauss5_LL" + end_name)
    # # data3 = np.load('./ablation/' + file_path + "/wrong_gauss5_LL" + end_name)
    data1 = np.load(file_path + filename1)
    data2 = np.load(file_path + filename2)

    # data3 = np.load('./fidelity' + file_path + filename2 + '_test_acc.npy')
    # data4 = np.load('./fidelity' + file_path + filename2 + '_tri_acc.npy')

    # # 画出折线图并添加图例
    data1 = data1[:13] * 100
    data2 = data2[:13] * 100
    # data3 = data3[:16] * 100
    # data4 = data4[:16] * 100
    # # data1 = data1 * 100
    # # data2 = data2 * 100
    # # data3 = data3 * 100
    # # data4 = data4 * 100
    save_path = 'G:/2023春/论文/figures/ss1.svg'

    plot_graph(data1, data2, save_path)
    # plt.plot(data1, label='proposed')
    # plt.plot(data2, label='NLM')
    # plt.plot(data3, label='N2LM')
    #
    # # 添加x轴和y轴标签
    # plt.xlabel('epoch')
    # plt.ylabel('Acc')
    #
    # # 添加图例
    # plt.legend()
    #
    # # 显示图形
    # plt.show()

    # #画一个
    # np.random.seed(2000)
    # noise = np.random.randint(0, 60, size=(32, 280, 3))
    # plt.imshow(noise, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # a = np.load('955_psnr.npy')
    # b = np.load('955_ssim.npy')
    # c = np.load('955_lpips.npy')
    # print(a.mean(), b.mean(), c.mean())
    # print(a.var(), b.var(), c.var())

    # triggers_filename = './crnn/data/part_300w/txt/gtrigger.txt'
    # show_photos(triggers_filename)