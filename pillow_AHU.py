from PIL import Image
import numpy as np
import cv2
import os
import time
import numpy as np

pwd = os.path.abspath(os.curdir)  # 当前文件夹：./data/
print('当前文件夹：', pwd)
data_set_list = os.listdir('./data/Street')  # 所有数据文件夹列表：00 01 02 ...
# data_set_list.remove('hehe.py')  # 删除该文件本身
# os.mkdir('./test')  # 创建目标文件夹
print('所有图片文件夹列表：', data_set_list)


def cv_resize_image(path, image):
    image_path = path+'/'+image
    print("image_path =", image_path)
    img = Image.open(image_path)
    img_npy = np.array(img)
    new_img_npy = img_npy[112: 400, :, :]
    new_img = Image.fromarray(new_img_npy).convert('RGB')
    img_cv = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
    new_image = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("new_image", new_image)
    # cv2.waitKey()
    return new_image

# cnt = 0
for data_set in data_set_list:
    # print(data_set)
    path = pwd + '/data/Street/' + data_set  # ./data/train/drone/00
    print('path =', path)
    image_set = os.listdir(path)  # 图片列表：image1 image2 ...
    print('图片列表：', len(image_set))
    for image in image_set:
        print(image)
        new_image = cv_resize_image(path, image)
        obj_dir = '/home/biyisi/PycharmProjects/pythonProject/data/new_street/' + data_set + '/'
        print("obj_dir =", obj_dir)
        cv2.imwrite(obj_dir+image, new_image)
        time.sleep(0.1)

    # cnt += len(image_set)
# print(cnt)


















        # time.sleep(1000)
    # # 随机取1/5
    # test_set = np.random.choice(image_set, size=int(len(image_set) / 5))
    # print('作为test的图片列表：', len(test_set))
    # # # 创建相应的test文件夹
    # obj_dir = '~/图片/AHU/test/drone/' + data_set
    # print('成功创建文件夹：', obj_dir)
    # # os.mkdir(obj_dir)
    # os.system('mkdir %s' % obj_dir)
    # # # 开始移动文件
    # for test in test_set:
    #     cmd = 'mv ' + path + '/' + test + ' ' + obj_dir + '/'  # 指令
    #     os.system(cmd)
    #     print('正在执行：', cmd)


# def image_open():
#     image_path = "./data/train/drone/00/image_1.jpg"
#     image = Image.open(image_path)
#     print("image_shape: ", image.size)
#     image.show()
#     return image
#
#
# # image_open()
# image: Image = image_open()
# image_npy = np.array(image)
#
# new_image_npy = image_npy[112: 400, :, :]
# # new_image_npy = np.resize(new_image_npy, (256, 256))
# new_image = Image.fromarray(new_image_npy).convert('RGB')
# # image = cv2.imread("./data/train/drone/00/image_1.jpg")
# img = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
# cv2.imshow('OriginalPicture', img)
# # cv2.waitKey()
# # new_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
# new_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('newPicture', new_image)
# cv2.waitKey()
















# new_image.show()

# for i in range(300, 512):
#     if image_npy[i][0][0] == 0:
#         print("i=", i)
#         break

# 0-111
# 400-511

# print(image_npy[100][1][0])

# print(type(image_npy))
# print(image_npy)
