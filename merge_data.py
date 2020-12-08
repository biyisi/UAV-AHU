import os
import sys
import time

# pwd = os.path.abspath(os.curdir)  # 当前文件夹：./data/
# print('当前文件夹：', pwd)
# data_set_list = os.listdir('./data/Street')  # 所有数据文件夹列表：00 01 02 ...
# print('所有图片文件夹列表：', data_set_list)

def merge_data(end_path):
    path = "/home/biyisi/PycharmProjects/pythonProject/data/train/drone/"+end_path
    print('path =', path)
    image_set = os.listdir(path)  # 图片列表：image1 image2 ...
    print('图片列表：', len(image_set))
    for image in image_set:
        print(image)
        obj_dir = '/home/biyisi/PycharmProjects/pythonProject/data/train/view/' + end_path + '/'
        print("obj_dir =", obj_dir)
        cmd = 'cp ' + path + '/' + image + ' ' + obj_dir + '/'  # 指令
        os.system(cmd)
        time.sleep(0.01)


merge_data("00")
merge_data("01")
merge_data("02")
merge_data("03")
merge_data("04")
merge_data("05")
merge_data("06")
merge_data("07")
merge_data("08")
merge_data("09")
merge_data("10")
merge_data("11")
merge_data("12")
merge_data("13")
merge_data("14")
merge_data("15")
merge_data("16")
merge_data("17")
merge_data("18")
merge_data("19")
merge_data("20")
