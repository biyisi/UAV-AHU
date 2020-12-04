import scipy.io
import torch
import numpy as np
# import time
import os


def evaluate(qf: torch.cuda.FloatTensor, ql, gf: torch.cuda.FloatTensor, gl):
    """
    @params:
    qf: query_feature (512)
    ql: query_label (1)
    gf: gallery_feature (51355x512)
    gl: gallery_label (51355)
    @returns:
    ap_tmp
    CMC_tmp
    """
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
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # find good_index index
    n_good = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
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
    result = scipy.io.loadmat('new_satellite->drone.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    print("type(query_feature)=", type(query_feature))

    query_label = result['query_label'][0]
    print("type(query_label)=", type(query_label))

    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # print(query_feature.shape)
    # print(gallery_feature.shape)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0  # ap衡量模型在每个类别上的好坏

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        # print("ap_tmp = ", ap_tmp)
        # print("CMC_tmp = ", CMC_tmp)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(round(len(gallery_label) * 0.01))
    print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
        CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100,
        ap / len(query_label) * 100))
