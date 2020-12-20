import numpy as np
import torch
import os
import time

'''
    从[0,20]之间取8个随机数
'''


# x = torch.randint(0, 21, size=(8, ))
# print(x)

def del_data(end_path, lendel: int):
    path = "/home/biyisi/PycharmProjects/pythonProject/data/train/view/" + end_path
    print('path =', path)
    image_set = os.listdir(path)  # 图片列表：image1 image2 ...
    print('图片列表：', len(image_set))
    cnt: int = 0
    for image in image_set:
        cnt = cnt + 1
        if cnt % lendel == 0:
            cmd = 'rm -rf ' + path + '/' + image
            print(cmd)
            os.system(cmd)
            time.sleep(0.01)
            # pass
        # print(image)
        # obj_dir = '/home/biyisi/PycharmProjects/pythonProject/data/train/view/' + end_path + '/'
        # print("obj_dir =", obj_dir)
        # cmd = 'cp ' + path + '/' + image + ' ' + obj_dir + '/'  # 指令
        # os.system(cmd)
        # time.sleep(0.01)


del_data('02', 2)
del_data('03', 2)
del_data('07', 2)
del_data('09', 2)
del_data('14', 2)
del_data('16', 2)
del_data('17', 2)
del_data('20', 2)
