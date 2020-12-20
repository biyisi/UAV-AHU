import time

import scipy.io
import torch
import numpy as np
# import time
import os


def evaluate(qf: torch.cuda.FloatTensor, ql, gf_temp: torch.cuda.FloatTensor, gl_temp, i: int):
    """
    @params:
    qf: query_feature (512)
    ql: query_label (1)
    gf: gallery_feature (2547x512)
    gl: gallery_label (2547)
    @returns:
    ap_tmp
    CMC_tmp
    """


    gf = gf_temp.clone()
    gf[i] = 0
    # gf = gf[torch.arange(gf.size(0)) != i]

    gl = gl_temp.copy()
    # gl = np.delete(gl, [i], axis=0)

    query = qf.view(-1, 1)  # 拉直(512x1维向量)
    # print(query.shape)
    score = torch.mm(gf, query)  # 矩阵乘法(shape=51355x1)
    score = score.squeeze(1).cpu()  # 去掉dim=1那一维(shape=51355)
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]  # from large to small
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl == -1)

    ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
    return ap_tmp, CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  # 判断index每个元素是否在junk_index中，返回bool ndarray
    index = index[mask]  # 取在junk_index中找到的部分
    # find good_index index
    n_good = len(good_index)
    mask = np.in1d(index, good_index)  # 再在good_index中找
    rows_good = np.argwhere(mask == True)  # 返回True的下标
    rows_good = rows_good.flatten()  # 拉直

    cmc[rows_good[0]:] = 1  # 第一个True之后都置为1
    for i in range(n_good):
        d_recall = 1.0 / n_good
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc


if __name__ == '__main__':
    # result = scipy.io.loadmat('drone->satellite.mat')
    # result = scipy.io.loadmat('satellite->drone.mat')
    # result = scipy.io.loadmat('drone->satellite_new.mat')

    result_logits_view = scipy.io.loadmat('CNN_100+_6798_label_view.mat')
    logits_feature = torch.FloatTensor(result_logits_view['query_feature'])
    logits_feature.cuda()
    logits_label = result_logits_view['query_label'][0]
    logits_path = result_logits_view['query_path']
    CMC_logits = torch.IntTensor(len(logits_label)).zero_()
    logits_AP = 0
    logits_len = len(logits_label)
    print("logits_len =", logits_len)

    for i in range(logits_len):
        logits_AP_temp, CMC_logits_temp = evaluate(logits_feature[i], logits_label[i], logits_feature,
                                                     logits_label, i)
        # print("logits_AP_temp =", logits_AP_temp)
        # print("CMC_logits_temp =", CMC_logits_temp)
        if CMC_logits_temp[0] == -1:
            continue
        CMC_logits = CMC_logits + CMC_logits_temp
        logits_AP = logits_AP + logits_AP_temp
        # print("i, CMC_teacher_temp[0] =", i, CMC_teacher_temp[0])

    CMC_logits = CMC_logits.float()
    CMC_logits = CMC_logits / logits_len
    print("round(logits_len * 0.01) =", round(logits_len * 0.01))
    print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC_logits[0] * 100, CMC_logits[4] * 100, CMC_logits[9] * 100, CMC_logits[round(logits_len * 0.01)] * 100,
        logits_AP / logits_len * 100))




    # result_teacher_view = scipy.io.loadmat('teacher_159_label_view.mat')
    # result_student_view = scipy.io.loadmat('student_159_label_view.mat')
    # result_kd_view = scipy.io.loadmat('kd_159_label_view.mat')

    # result_teacher_view = scipy.io.loadmat('teacher_6798_view_all.mat')
    # teacher_feature = torch.FloatTensor(result_teacher_view['query_feature'])
    # teacher_feature.cuda()
    # teacher_label = result_teacher_view['query_label'][0]
    # teacher_path = result_teacher_view['query_path']
    # CMC_teacher = torch.IntTensor(len(teacher_label)).zero_()
    # teacher_AP = 0
    # teacher_len = len(teacher_label)
    # # print("CMC_teacher =", CMC_teacher)
    # # print("teacher_feature =", teacher_feature)
    # # print("teacher_label =", teacher_label)
    # # print("teacher_path =", teacher_path)
    # print("teacher_len =", teacher_len)
    #
    # for i in range(teacher_len):
    #     teacher_AP_temp, CMC_teacher_temp = evaluate(teacher_feature[i], teacher_label[i], teacher_feature,
    #                                                  teacher_label, i)
    #     # print("teacher_AP_temp =", teacher_AP_temp)
    #     # print("CMC_teacher_temp =", CMC_teacher_temp)
    #     if CMC_teacher_temp[0] == -1:
    #         continue
    #     CMC_teacher = CMC_teacher + CMC_teacher_temp
    #     teacher_AP = teacher_AP + teacher_AP_temp
    #     # print("i, CMC_teacher_temp[0] =", i, CMC_teacher_temp[0])
    #
    # CMC_teacher = CMC_teacher.float()
    # CMC_teacher = CMC_teacher / teacher_len
    # print("round(teacher_len * 0.01) =", round(teacher_len * 0.01))
    # print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    #     CMC_teacher[0] * 100, CMC_teacher[4] * 100, CMC_teacher[9] * 100, CMC_teacher[round(teacher_len * 0.01)] * 100,
    #     teacher_AP / teacher_len * 100))



    # student_feature = torch.FloatTensor(result_student_view['query_feature'])
    # student_feature.cuda()
    # student_label = result_student_view['query_label'][0]
    # student_path = result_student_view['query_path']
    # CMC_student = torch.IntTensor(len(student_label)).zero_()
    # student_AP = 0
    # student_len = len(student_label)
    # # print("CMC_student =", CMC_student)
    # # print("student_feature =", student_feature)
    # # print("student_label =", student_label)
    # # print("stdent_label.shape =", student_label.shape)
    # # print("student_path =", student_path)
    # print("student_len =", student_len)
    # # print(type(student_feature))
    # # print(type(student_label))
    #
    # for i in range(student_len):
    #     student_AP_temp, CMC_student_temp = evaluate(student_feature[i], student_label[i], student_feature,
    #                                                  student_label, i)
    #     # print("teacher_AP_temp =", teacher_AP_temp)
    #     # print("CMC_teacher_temp =", CMC_teacher_temp)
    #     if CMC_student_temp[0] == -1:
    #         continue
    #     CMC_student = CMC_student + CMC_student_temp
    #     student_AP = student_AP + student_AP_temp
    #     # print("i, CMC_teacher_temp[0] =", i, CMC_teacher_temp[0])
    #
    # CMC_student = CMC_student.float()
    # CMC_student = CMC_student / student_len
    # print("round(student_len * 0.01) =", round(student_len * 0.01))
    # print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    #     CMC_student[0] * 100, CMC_student[4] * 100, CMC_student[9] * 100, CMC_student[round(student_len * 0.01)] * 100,
    #     student_AP / student_len * 100))


    # kd_feature = torch.FloatTensor(result_kd_view['query_feature'])
    # kd_feature.cuda()
    # kd_label = result_kd_view['query_label'][0]
    # kd_path = result_kd_view['query_path']
    # CMC_kd = torch.IntTensor(len(kd_label)).zero_()
    # kd_AP = 0
    # kd_len = len(kd_label)
    # # print("CMC_kd =", CMC_kd)
    # # print("kd_feature =", kd_feature)
    # # print("kd_label =", kd_label)
    # # print("kd_path =", kd_path)
    # print("kd_len =", kd_len)
    #
    # for i in range(kd_len):
    #     kd_AP_temp, CMC_kd_temp = evaluate(kd_feature[i], kd_label[i], kd_feature,
    #                                                  kd_label, i)
    #     # print("teacher_AP_temp =", teacher_AP_temp)
    #     # print("CMC_teacher_temp =", CMC_teacher_temp)
    #     if CMC_kd_temp[0] == -1:
    #         continue
    #     CMC_kd = CMC_kd + CMC_kd_temp
    #     kd_AP = kd_AP + kd_AP_temp
    #     # print("i, CMC_teacher_temp[0] =", i, CMC_teacher_temp[0])
    #
    # CMC_kd = CMC_kd.float()
    # CMC_kd = CMC_kd / kd_len
    # print("round(kd_len * 0.01) =", round(kd_len * 0.01))
    # print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    #     CMC_kd[0] * 100, CMC_kd[4] * 100, CMC_kd[9] * 100, CMC_kd[round(kd_len * 0.01)] * 100,
    #     kd_AP / kd_len * 100))


    # CMC = CMC / len(query_label)  # average CMC
    # print(round(len(gallery_label) * 0.01))
    # print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    #     CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
    #     ap / len(query_label) * 100))

# for i in range(len(query_label)):
#     ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
#     # print("ap_tmp = ", ap_tmp)
#     # print("CMC_tmp = ", CMC_tmp)
#     if CMC_tmp[0] == -1:
#         continue
#     CMC = CMC + CMC_tmp
#     ap += ap_tmp
#     # print(i, CMC_tmp[0])

# query_feature = torch.FloatTensor(result['query_f'])
# print("type(query_feature)=", type(query_feature))
#
# query_label = result['query_label'][0]
# print("type(query_label)=", type(query_label))
#
# gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_label = result['gallery_label'][0]
#
# query_feature = query_feature.cuda()
# gallery_feature = gallery_feature.cuda()

# print(query_feature.shape)
# print(gallery_feature.shape)

# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0  # ap衡量模型在每个类别上的好坏
#

#
# CMC = CMC.float()
# CMC = CMC / len(query_label)  # average CMC
# print(round(len(gallery_label) * 0.01))
# print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
#     CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
#     ap / len(query_label) * 100))
