import os
import sys


# pwd = os.path.abspath(os.curdir)  # 当前文件夹：./data/
# print('当前文件夹：', pwd)
# data_set_list = os.listdir('./data/Street')  # 所有数据文件夹列表：00 01 02 ...
# print('所有图片文件夹列表：', data_set_list)

def rename(end_path):
    # path = "/home/biyisi/PycharmProjects/pythonProject/data/test/street/"+end_path
    path = "/home/biyisi/图片/1206/"+end_path
    count = 1
    filelist = os.listdir(path)
    for file in filelist:
        oldFile = os.path.join(path, file)
        if os.path.isfile(oldFile):
            newFile = os.path.join(path, "1206_"+end_path+"_street_" + str(count) + ".jpg")
            os.rename(oldFile, newFile)
        else:
            continue
        count += 1
    print("一共修改了" + str(count) + "个文件")

rename('19')
# rename("00")
# rename("01")
# rename("02")
# rename("03")
# rename("04")
# rename("05")
# rename("06")
# rename("07")
# rename("08")
# rename("09")
# rename("10")
# rename("11")
# rename("12")
# rename("13")
# rename("14")
# rename("15")
# rename("16")
# rename("17")
# rename("18")
# rename("19")
# rename("20")
