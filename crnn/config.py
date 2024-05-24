from pprint import pprint
import torch
class Config:
    # data

    # 300w
    # train_filename = 'G:/BaiduNetdiskDownload/Synthetic Chinese String Dataset/train.txt'
    # val_filename = 'G:/BaiduNetdiskDownload/Synthetic Chinese String Dataset/test.txt'
    # root_dir = 'G:/BaiduNetdiskDownload/Synthetic Chinese String Dataset/images'
    # part 300w
    train_filename = './crnn/data/part_300w/txt/train45000.txt'
    val_filename = './crnn/data/part_300w/txt/val5000.txt'
    root_dir = 'G:/BaiduNetdiskDownload/Synthetic Chinese String Dataset/images'
    wrong_filename = './crnn/data/part_300w/txt/all_wrong_samples.txt'


    char_dict_file = './crnn/data/part_300w/txt/m_char_std_5990.txt'
    image_size = (32, 280)
    max_label_length = 10

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # network
    nclasses = 5990

    # training
    epoch = 100

    # model
    # load_path = "./crnn/trained_weights/epoch_2_epoch_loss0.00045_time_Wed Feb 27 10_13_23 2019.pt"
    load_path = "./crnn/trained_weights/epoch_99_loss0.00849_time_1675807548.9743123.pt"
    trained_weights = './crnn/epoch_0_epoch_loss0.02102_time_Tue Feb 26 21:45:24 2019.pt'
    # watermarked_weights = "./crnn/watermarked_weights/fragile_36random_watermarking_model_1e-4_newent.pth"
    # watermarked_weights = "./crnn/watermarked_weights/fragile_36random_watermarking_model.pth"
    file_path = "/36_0.003_1e-4"
    file_name = "/wrong_gauss5_LL"
    watermarked_weights1 = "./crnn/watermarked_weights" + file_path + file_name + ".pth"
    watermarked_weights3 = "./crnn/watermarked_weights/fragile_36gauss_singlewm" + "1e-4" + ".pth"
    watermarked_weights2 = './crnn/finetuned_weights/lsb_0.01_weights.pth'
    watermarked_weights = "./crnn/watermarked_weights/fragile_36wrong_singlewm_nos2_normal.pth"
    # watermarked_weights = "./crnn/watermarked_weights/fragile_36wrong_watermarking_1e-4_newent.pth"
    pruning_weights = "./crnn/pruned_weights/pruning0.5_36random_watermarking_model2.pth"
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('Unknow Option: "--%s"' % k)
            setattr(self, k, v)
        print('**********************************user config*************************')
        pprint(self._state_dict())
        print('*************************************end******************************')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}

opt = Config()
