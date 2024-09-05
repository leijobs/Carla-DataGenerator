import os
import random
import shutil

import numpy as np

if __name__ == "__main__":
    dataset_dir = r'/home/hosico/DataDisk/hdd2/Dataset/VKITTI/DetSeg'
    train_dir = r'training'
    test_dir = r'testing'

    # path_list = ['calib', 'gnss', 'image', 'image_seg', 'imu', 'label', 'velodyne']
    # for path_info in path_list:
    #     test_abs = os.path.join(dataset_dir, test_dir, path_info)
    #     if not os.path.exists(test_abs):
    #         print("mkdir : ", test_abs)
    #         os.mkdir(test_abs)
    path_map = {
        'calib': '.txt',
        'gnss': '.txt',
        'image': '.png',
        'image_seg': '.png',
        'imu': '.txt',
        'label': '.txt',
        'velodyne': '.bin'}
    # test_list = random.sample(range(0, 15001), 7518)
    # for path_info, ext_info in path_map.items():
    #     print("processing  ", path_info)
    #     for test_file in test_list:
    #         test_name = str(test_file).zfill(6) + ext_info
    #         train_abs = os.path.join(dataset_dir, train_dir, path_info, test_name)
    #         test_abs = os.path.join(dataset_dir, test_dir, path_info, test_name)
    #         print('moving .. : ', train_abs, '  into  ', test_abs)
    #         shutil.move(train_abs, test_abs)

    test_abs = os.path.join(dataset_dir, train_dir, 'gnss')
    test_list = os.listdir(test_abs)

    test_seq = [data.split('.txt')[0] for data in test_list]
    test_seq.sort()
    test_txt = r'/home/hosico/DataDisk/hdd2/Dataset/VKITTI/DetSeg/training.txt'

    with open(test_txt, 'w') as file:
        for item in test_seq:
            file.write(item + '\n')
    print(test_seq)
