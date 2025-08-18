from torch.autograd.variable import Variable
import pickle
from collections import Counter
# from train_old_version2 import train, evaluate
import warnings
# from cross_valiad import cross_val
# from proprecess import divide, cal_aal_pcc, cal_aal_partial, preprocess_corr
# from model_version import final_model
# from load_data import load_data
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from pytorchtools import EarlyStopping
import time
import math
from torch.nn import init
import pandas as pd
import os
import numpy as np
import torch
import random
from Graph_sample import datasets2, datasets2_train, datasets2_train_original, datasets2_train_supervised
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter
import argparse

def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--data_path', type=str,
                        default='/home/linux/brain/cc200/',
                        help='path to the data')
    parser.add_argument('--label_path', type=str,
                        default=r'/home/linux/brain/Phenotypic_V1_0b_preprocessed1.csv',
                        help='path to the label')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--split_num', type=int, default=5, metavar='S', help='number of K split')
    parser.add_argument('--ratio_outlier', type=int, default=0.08, metavar='S', help='ratio of labelled data')
    parser.add_argument('--log_dir', type=str, default=r'MTL_0.9_noiseup_protype_supervised', help='path of tensorboard')
    parser.add_argument('--num_workers', type=int, default=4, metavar='S', help='number of cpu workers')
    parser.add_argument('--epochs', type=int, default=500, metavar='S', help='number of epochs')
    parser.add_argument('--epochs_original', type=int, default=300, metavar='S', help='number of original epochs')
    parser.add_argument('--epochs_early', type=int, default=200, metavar='S', help='number of early epochs')
    parser.add_argument('--epochs_middle', type=int, default=300, metavar='S', help='number of middle epochs')
    parser.add_argument('--epochs_late', type=int, default=500, metavar='S', help='number of late epochs')
    parser.add_argument('--w_early', type=int, default=0.1, metavar='S', help='weight of early')
    parser.add_argument('--w_middle', type=int, default=0.3, metavar='S', help='weight of middle')
    parser.add_argument('--w_late', type=int, default=0.6, metavar='S', help='weight of late')
    parser.add_argument('--batch_size', type=int, default=128, metavar='S', help='number of batch_size')
    parser.add_argument('--phi', type=int, default=0.6, metavar='S', help='phi for cal_pcc')
    parser.add_argument('--phi1', type=int, default=0.4, metavar='S', help='phi for cal_pcc')
    parser.add_argument('--phi2', type=int, default=0.65, metavar='S', help='phi for cal_pcc')
    parser.add_argument('--super_nodes', type=int, default=8, metavar='S', help='number of super nodes')
    parser.add_argument('--wp', type=int, default=1, metavar='S', help='weights for loss_u')
    parser.add_argument('--threshold', type=int, default=0.95, metavar='S', help='threshold for unlabelled data')

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

args = get_args()


acc_1 = []
acc_2 = []
acc_3 = []
acc_4 = []
acc_5 = []

# prec_1 = []
# prec_2 = []
# prec_3 = []
# prec_4 = []
# prec_5 = []
#
# recall_1 = []
# recall_2 = []
# recall_3 = []
# recall_4 = []
# recall_5 = []
#
# F1_1 = []
# F1_2 = []
# F1_3 = []
# F1_4 = []
# F1_5 = []
#
# auc_1 = []
# auc_2 = []
# auc_3 = []
# auc_4 = []
# auc_5 = []
log_dir = args.log_dir



# 如果目录不存在，则创建目录
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer1 = SummaryWriter(log_dir=log_dir)
###############################################################
###############################################################
# 导入数据阶段
dim = 200
path1 = args.data_path
path2 = args.label_path


def uncertainty(output):
    output_tensor = torch.from_numpy(output)
    probabilities = torch.softmax(output_tensor, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
    return entropy


def ema_update(teacher, student, ema_decay, cur_step=None):
    if cur_step is not None:
        ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)


def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key


'''
导入数据进来
'''


def load_data(path1, path2):
    profix = path1
    dirs = os.listdir(profix)
    all = {}
    labels = {}
    all_data = []
    label = []
    files = open(r'/home/linux/brain/files.txt', 'r', encoding='utf-8')
    for filename in dirs:
        filename = files.readline().strip()
        print(filename)
        a = np.loadtxt(path1 + filename)
        # print(filename)
        a = a.transpose()
        # a = a.tolist()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if get_key(filename) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 1:
                    labels[filename] = int(data['DX_GROUP'][i])
                    label.append(int(data['DX_GROUP'][i]))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    return all, labels, all_data, label  # 871 * 116 * ?


def cal_aal_pcc(time_data):
    corr_matrix = []
    for key in range(len(time_data)):
        corr = []
        for key1 in range(len(time_data[key])):
            corr_mat = np.corrcoef(time_data[key][key1])
            corr.append(corr_mat)
        corr_matrix.append(corr)
    data_array = np.array(corr_matrix)
    where_are_nan = np.isnan(data_array)  # 找出数据中为nan的
    where_are_inf = np.isinf(data_array)  # 找出数据中为inf的
    for bb in range(0, 871):
        for k in range(data_array.shape[1]):
            for i in range(0, dim):
                for j in range(0, dim):
                    if where_are_nan[bb][k][i][j]:
                        data_array[bb][k][i][j] = 0
                    if where_are_inf[bb][k][i][j]:
                        data_array[bb][k][i][j] = 0.8
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 4 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3, 4))
    return data_array

def cal_pcc_noise(length_raw, data, phi):
    '''
    :param data:  图   871 * 116 * ?
    :return:  adj
    '''
    corr_matrix = []
    for key in range(len(data)):  # 每一个sample
        corr_mat = np.corrcoef(data[key])
        # if key == 5:
        #    print(corr_mat)
        corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))

        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)  # 871 116 116

    # #加数据增强
    noise = np.random.normal(loc=0, scale=0.3, size=data_array.shape)


    data_array = data_array + noise
    # data_array = np.concatenate([data_array, data_array_noise], axis=0)
    # print(data_array.shape)
    # print(1/0)

    # print("data_array", data_array)
    where_are_nan = np.isnan(data_array)  # 找出数据中为nan的
    where_are_inf = np.isinf(data_array)  # 找出数据中为inf的
    # for bb in range(0, length_raw):
    #     for i in range(0, dim):
    #         for j in range(0, dim):
    #             if where_are_nan[bb][i][j]:
    #                 data_array[bb][i][j] = 0
    #             if where_are_inf[bb][i][j]:
    #                 data_array[bb][i][j] = 1
    #             if data_array[bb][i][j] > phi:
    #                 data_array[bb][i][j] = 1
    #             elif data_array[bb][i][j] < phi*(-1):
    #                 data_array[bb][i][j] = -1
    #             else:
    #                 data_array[bb][i][j] = 0
    data_array[where_are_nan] = 0
    data_array[where_are_inf] = 1
    data_array[data_array > phi] = 1
    data_array[data_array < -phi] = -1
    data_array[(data_array >= -phi) & (data_array <= phi)] = 0
    # print(data_array[0])
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3))

    return data_array


def cal_pcc(data, phi):
    '''
    :param data:  图   871 * 116 * ?
    :return:  adj
    '''
    corr_matrix = []
    for key in range(len(data)):  # 每一个sample
        corr_mat = np.corrcoef(data[key])
        # if key == 5:
        #    print(corr_mat)
        corr_mat = np.arctanh(corr_mat - np.eye(corr_mat.shape[0]))

        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)  # 871 116 116
    where_are_nan = np.isnan(data_array)  # 找出数据中为nan的
    where_are_inf = np.isinf(data_array)  # 找出数据中为inf的
    # for bb in range(0, 871):
    #     for i in range(0, dim):
    #         for j in range(0, dim):
    #             if where_are_nan[bb][i][j]:
    #                 data_array[bb][i][j] = 0
    #             if where_are_inf[bb][i][j]:
    #                 data_array[bb][i][j] = 1
    #             if data_array[bb][i][j] > phi:
    #                 data_array[bb][i][j] = 1
    #             elif data_array[bb][i][j] < phi*(-1):
    #                 data_array[bb][i][j] = -1
    #             else:
    #                 data_array[bb][i][j] = 0
    data_array[where_are_nan] = 0
    data_array[where_are_inf] = 1
    data_array[data_array > phi] = 1
    data_array[data_array < -phi] = -1
    data_array[(data_array >= -phi) & (data_array <= phi)] = 0
    # print(data_array[0])
    corr_p = np.maximum(data_array, 0)  # pylint: disable=E1101
    corr_n = 0 - np.minimum(data_array, 0)  # pylint: disable=E1101
    data_array = [corr_p, corr_n]
    data_array = np.array(data_array)  # 2 871 116 116
    data_array = np.transpose(data_array, (1, 0, 2, 3))

    return data_array


###############################################################################
###############################################################################

import scipy.io as sio
import scipy.sparse as sp
from sklearn import preprocessing
#################################################################################
###############################################################################
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().A
def normalize_adj_new(adj):
    """Symmetrically normalize adjacency matrix."""
    A_list = []
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            A_list.append(adj[i][j])
    print(max(A_list))
    A_normalized = preprocessing.normalize(np.array(A_list)[:, np.newaxis], axis=0).ravel()
    # A = np.array(A_list)
    # A_normalized = A / np.linalg.norm(A)
    idx = 0
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            adj[i][j] = A_normalized[idx]
            adj[j][i] = A_normalized[idx]
            idx+=1
    return adj
    # adj = sp.coo_matrix(adj)
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().A
# 数据集划分

# def cross_val_semi(selected_indices_list, raw_data, A, A1, A2, labels):
#     kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
#     zip_list = list(zip(raw_data, A, A1, A2, labels))
#     # 这里的随机打乱我的代码里没有
#     random.Random(0).shuffle(zip_list)
#     raw_data, A, A1, A2, labels = zip(*zip_list)
#
#     test_data_loader = []
#     train_data_loader = []
#     valid_data_loader = []
#     raw_data = np.array(raw_data)
#     A = np.array(A)
#     A1 = np.array(A1)
#     A2 = np.array(A2)
#     labels = np.array(labels)
#     for kk, (train_index, test_index) in enumerate(kf.split(A, labels)):
#
#         train_val_raw_data = raw_data[train_index]
#
#         train_val_adj, test_adj = A[train_index], A[test_index]
#         train_val_adj1, test_adj1 = A1[train_index], A1[test_index]
#         train_val_adj2, test_adj2 = A2[train_index], A2[test_index]
#         train_val_labels, test_labels = labels[train_index], labels[test_index]
#
#         train_val_adj_labelled = np.array([train_val_adj[idx] for idx in range(train_index.shape[0]) if idx not in selected_indices_list[0]])
#         train_val_adj_labelled1 = np.array([train_val_adj1[idx] for idx in range(train_index.shape[0]) if
#                                   idx not in selected_indices_list[0]])
#         train_val_adj_labelled2 = np.array([train_val_adj2[idx] for idx in range(train_index.shape[0]) if
#                                    idx not in selected_indices_list[0]])
#
#         train_val_labels_labelled = np.array([train_val_labels[idx] for idx in range(train_index.shape[0]) if
#                                    idx not in selected_indices_list[0]])
#
#         raw_labelled = np.array([train_val_raw_data[idx] for idx in range(train_index.shape[0]) if
#                                      idx not in selected_indices_list[0]])
#
#         dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
#         test_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=0)
#         test_data_loader.append(test_dataset_loader)
#
#         dataset_sampler = datasets2_train_supervised(raw_labelled, train_val_adj_labelled, train_val_adj_labelled1, train_val_adj_labelled2, train_val_labels_labelled)
#         train_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=0)
#         train_data_loader.append(train_dataset_loader)
#         dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
#         val_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=0)
#         valid_data_loader.append(val_dataset_loader)
#
#     return train_data_loader, valid_data_loader, test_data_loader
#
# def cross_val(raw_data, A, A1, A2, labels):
#     kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
#     zip_list = list(zip(raw_data, A, A1, A2, labels))
#     # 这里的随机打乱我的代码里没有
#     random.Random(0).shuffle(zip_list)
#     raw_data, A, A1, A2, labels = zip(*zip_list)
#
#     test_data_loader = []
#     train_data_loader = []
#     valid_data_loader = []
#     raw_data = np.array(raw_data)
#     A = np.array(A)
#     A1 = np.array(A1)
#     A2 = np.array(A2)
#     labels = np.array(labels)
#     for kk, (train_index, test_index) in enumerate(kf.split(A, labels)):
#
#         train_val_adj, test_adj = A[train_index], A[test_index]
#         train_val_adj1, test_adj1 = A1[train_index], A1[test_index]
#         train_val_adj2, test_adj2 = A2[train_index], A2[test_index]
#         train_val_labels, test_labels = labels[train_index], labels[test_index]
#
#
#
#         dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
#         test_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=0)
#         test_data_loader.append(test_dataset_loader)
#
#         # dataset_sampler = datasets2_train(raw_labelled, raw_unlabelled, train_val_adj_labelled, train_val_adj_labelled1, train_val_adj_labelled2, train_val_labels_labelled, train_val_adj_unlabelled, train_val_adj_unlabelled1, train_val_adj_unlabelled2)
#         dataset_sampler = datasets2_train_original(train_val_adj, train_val_adj1, train_val_adj2, train_val_labels)
#         train_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=0)
#         train_data_loader.append(train_dataset_loader)
#         dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
#         val_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=0)
#         valid_data_loader.append(val_dataset_loader)
#
#     return train_data_loader, valid_data_loader, test_data_loader


from sklearn.model_selection import StratifiedKFold, train_test_split

def cross_val_semi(selected_indices_list, raw_data, A, A1, A2, labels):
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    zip_list = list(zip(raw_data, A, A1, A2, labels))
    random.Random(0).shuffle(zip_list)
    raw_data, A, A1, A2, labels = zip(*zip_list)

    train_data_loader, valid_data_loader, test_data_loader = [], [], []

    raw_data, A, A1, A2, labels = map(np.array, (raw_data, A, A1, A2, labels))

    for kk, (train_val_index, test_index) in enumerate(kf.split(A, labels)):

        train_val_raw_data = raw_data[train_val_index]
        train_val_adj, test_adj = A[train_val_index], A[test_index]
        train_val_adj1, test_adj1 = A1[train_val_index], A1[test_index]
        train_val_adj2, test_adj2 = A2[train_val_index], A2[test_index]
        train_val_labels, test_labels = labels[train_val_index], labels[test_index]

        train_sub_idx, val_idx = train_test_split(
            np.arange(len(train_val_labels)),
            test_size=0.2,
            stratify=train_val_labels,
            random_state=kk
        )


        train_val_adj_labelled = np.array([train_val_adj[i] for i in train_sub_idx if i not in selected_indices_list[0]])
        train_val_adj_labelled1 = np.array([train_val_adj1[i] for i in train_sub_idx if i not in selected_indices_list[0]])
        train_val_adj_labelled2 = np.array([train_val_adj2[i] for i in train_sub_idx if i not in selected_indices_list[0]])
        train_val_labels_labelled = np.array([train_val_labels[i] for i in train_sub_idx if i not in selected_indices_list[0]])
        raw_labelled = np.array([train_val_raw_data[i] for i in train_sub_idx if i not in selected_indices_list[0]])

        dataset_sampler = datasets2_train_supervised(
            raw_labelled,
            train_val_adj_labelled,
            train_val_adj_labelled1,
            train_val_adj_labelled2,
            train_val_labels_labelled
        )
        train_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train_data_loader.append(train_loader)


        dataset_sampler = datasets2(
            train_val_adj[val_idx],
            train_val_adj1[val_idx],
            train_val_adj2[val_idx],
            train_val_labels[val_idx]
        )
        val_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=False, num_workers=0)
        valid_data_loader.append(val_loader)


        dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
        test_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_data_loader.append(test_loader)

    return train_data_loader, valid_data_loader, test_data_loader


def cross_val(raw_data, A, A1, A2, labels):
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    zip_list = list(zip(raw_data, A, A1, A2, labels))
    random.Random(0).shuffle(zip_list)
    raw_data, A, A1, A2, labels = zip(*zip_list)

    train_data_loader, valid_data_loader, test_data_loader = [], [], []

    raw_data, A, A1, A2, labels = map(np.array, (raw_data, A, A1, A2, labels))

    for kk, (train_val_index, test_index) in enumerate(kf.split(A, labels)):

        train_val_adj, test_adj = A[train_val_index], A[test_index]
        train_val_adj1, test_adj1 = A1[train_val_index], A1[test_index]
        train_val_adj2, test_adj2 = A2[train_val_index], A2[test_index]
        train_val_labels, test_labels = labels[train_val_index], labels[test_index]

        train_sub_idx, val_idx = train_test_split(
            np.arange(len(train_val_labels)),
            test_size=0.2,
            stratify=train_val_labels,
            random_state=kk
        )


        dataset_sampler = datasets2_train_original(
            train_val_adj[train_sub_idx],
            train_val_adj1[train_sub_idx],
            train_val_adj2[train_sub_idx],
            train_val_labels[train_sub_idx]
        )
        train_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train_data_loader.append(train_loader)


        dataset_sampler = datasets2(
            train_val_adj[val_idx],
            train_val_adj1[val_idx],
            train_val_adj2[val_idx],
            train_val_labels[val_idx]
        )
        val_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=False, num_workers=0)
        valid_data_loader.append(val_loader)


        dataset_sampler = datasets2(test_adj, test_adj1, test_adj2, test_labels)
        test_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_data_loader.append(test_loader)

    return train_data_loader, valid_data_loader, test_data_loader


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN, self).__init__()
        self.in_dim = in_dim  # 输入的维度
        self.out_dim = out_dim  # 输出的维度
        self.neg_penalty = neg_penalty  # 负值
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # GCN-node
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()  # 生成对角矩阵 feature_dim * feature_dim
        if x is None:  # 如果没有初始特征
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # I_cAXW = eye+self.c*AXW
        I_cAXW = eye + self.c * AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)  # 正则化后的值
        output = torch.nn.functional.softplus(y_norm)
        # output = y_relu
        # 做个尝试
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        return output


class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        ###################################
        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_1 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_1 = GCN(in_dim, hidden_dim, 0.2)
        # ----------------------------------
        self.gcn1_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_p_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn1_n_2 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n_2 = GCN(hidden_dim, hidden_dim, 0.2)
        # self.gcn3_n_2 = GCN(in_dim, hidden_dim, 0.2)
        # ---------------------------------
        self.gcn_p_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_p_shared1 = GCN(in_dim, hidden_dim, 0.2)
        self.gcn_n_shared1 = GCN(in_dim, hidden_dim, 0.2)
        # --------------------------------- ATT score
        self.Wp_1 = nn.Linear(self.hidden_dim, 1)
        self.Wp_2 = nn.Linear(self.hidden_dim, 1)
        self.Wp_3 = nn.Linear(self.hidden_dim, 1)
        self.Wn_1 = nn.Linear(self.hidden_dim, 1)
        self.Wn_2 = nn.Linear(self.hidden_dim, 1)
        self.Wn_3 = nn.Linear(self.hidden_dim, 1)
        ###################################
        self.kernel_p = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p1 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n1 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        self.kernel_p2 = nn.Parameter(torch.FloatTensor(dim, in_dim))  #
        self.kernel_n2 = nn.Parameter(torch.FloatTensor(dim, in_dim))
        # print(self.kernel_p)
        # self.kernel_p = Variable(torch.randn(116, 5)).cuda()  # 116 5
        # self.kernel_n = Variable(torch.randn(116, 5)).cuda()   # 116 5
        ################################################
        self.lin1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.lin1_1 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_1 = nn.Linear(16, self.out_dim)
        self.lin1_2 = nn.Linear(2 * in_dim * in_dim, 16)
        self.lin2_2 = nn.Linear(16, self.out_dim)
        self.losses = []
        self.losses1 = []
        self.losses2 = []
        self.mseLoss = nn.MSELoss()
        self.reset_weigths()
        self.nums = 3
        # 1 666 3 663
        with open(r'/home/linux/brain/regions2.txt', 'r') as f:
            counts = 0
            tmp_list = []
            for line in f:  # 116
                if counts == 0:
                    counts += 1
                    continue
                tmp = np.zeros(self.nums)
                line.strip('\n')
                line = line.split()

                for columns in range(self.nums):
                    # if columns != 2:
                    #     break
                    tmp[columns] = line[columns]

                tmp_list.append(tmp)
                counts += 1

        self.R = np.array(tmp_list).transpose((1, 0))
        self.R = torch.FloatTensor(self.R)
        self.ij = []
        print(self.R.shape)  # 6*116
        for ri in range(self.nums):
            tmp_sum = 0
            temp = []
            for i in range(dim):
                for j in range(i + 1, dim):
                    if self.R[ri][i] != 0 and self.R[ri][j] != 0:
                        temp.append((i, j))
            self.ij.append(temp)

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel, tell=None):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if tell == 'A':
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
                ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
                self.losses.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                          torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
                self.losses.append(neg_loss)
            self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        elif tell == 'A1':
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
                ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
                self.losses1.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses1.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                          torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
                self.losses1.append(neg_loss)
            self.losses1.append(0.05 * torch.sum(torch.abs(kernel_p)))
        elif tell == 'A2':
            if ortho_penalty != 0:
                ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
                ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
                self.losses2.append(ortho_loss)

            if variance_penalty != 0:
                variance = diag_elements.var()
                variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
                self.losses2.append(variance_loss)

            if neg_penalty != 0:
                neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                          torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
                self.losses2.append(neg_loss)
            self.losses2.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights
            """
        stdv = 1.0 / math.sqrt(dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, A, A1, A2):
        ##############################3
        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        A1 = torch.transpose(A1, 1, 0)
        s_feature_p1 = A1[0]
        s_feature_n1 = A1[1]
        A2 = torch.transpose(A2, 1, 0)
        s_feature_p2 = A2[0]
        s_feature_n2 = A2[1]
        ###############################

        ###############################
        p_reduce = self.dim_reduce(s_feature_p, 10, 0.2, 0.3, 0.1, self.kernel_p, tell='A')
        p_conv1_1_shared = self.gcn_p_shared(None, p_reduce) # shared GCN
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv1 = p_conv1 + p_conv1_1_shared # sum
        # p_conv1 = torch.cat((p_conv1, p_conv1_1_shared), -1) # concat
        p_conv2_1_shared = self.gcn_p_shared1(p_conv1, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        # p_conv2_1_shared = self.gcn_p_shared1(p_conv2, p_reduce)
        p_conv2 = p_conv2 + p_conv2_1_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce = self.dim_reduce(s_feature_n, 10, 0.2, 0.5, 0.1, self.kernel_n, tell='A')
        n_conv1_1_shared = self.gcn_n_shared(None, n_reduce) # shared GCN
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv1 = n_conv1 + n_conv1_1_shared # sum
        # n_conv1 = torch.cat((n_conv1, n_conv1_1_shared), -1) # concat
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        # n_conv2_1_shared = self.gcn_n_shared1(n_conv1, n_reduce)
        n_conv2 = n_conv2 + n_conv2_1_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce1 = self.dim_reduce(s_feature_p1, 10, 0.2, 0.3, 0.1, self.kernel_p1, tell='A1')
        p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1) # shared GCN
        p_conv1_1 = self.gcn1_p_1(None, p_reduce1)
        p_conv1_1 = p_conv1_1 + p_conv1_2_shared # sum
        # p_conv1_1 = torch.cat((p_conv1_1, p_conv1_2_shared), -1) # concat
        p_conv2_1 = self.gcn2_p_1(p_conv1_1, p_reduce1)
        p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        # p_conv2_2_shared = self.gcn_p_shared1(p_conv1_1, p_reduce1)
        p_conv2_1 = p_conv2_1 + p_conv2_2_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce1 = self.dim_reduce(s_feature_n1, 10, 0.2, 0.5, 0.1, self.kernel_n1, tell='A1')
        n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1) # shared GCN
        n_conv1_1 = self.gcn1_n_1(None, n_reduce1)
        n_conv1_1 = n_conv1_1 + n_conv1_2_shared # sum
        # n_conv1_1 = torch.cat((n_conv1_1, n_conv1_2_shared), -1) #concat
        n_conv2_1 = self.gcn2_n_1(n_conv1_1, n_reduce1)
        n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        # n_conv2_2_shared = self.gcn_n_shared1(n_conv1_1, n_reduce1)
        n_conv2_1 = n_conv2_1 + n_conv2_2_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ---------------------------------
        p_reduce2 = self.dim_reduce(s_feature_p2, 10, 0.2, 0.3, 0.1, self.kernel_p2, tell='A2')
        p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2) # shared GCN
        p_conv1_2 = self.gcn1_p_2(None, p_reduce2)
        p_conv1_2 = p_conv1_2 + p_conv1_3_shared # sum
        # p_conv1_2 = torch.cat((p_conv1_2, p_conv1_3_shared), -1) # concat
        p_conv2_2 = self.gcn2_p_2(p_conv1_2, p_reduce2)
        p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        # p_conv2_3_shared = self.gcn_p_shared1(p_conv1_2, p_reduce2)
        p_conv2_2 = p_conv2_2 + p_conv2_3_shared
        # p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce2 = self.dim_reduce(s_feature_n2, 10, 0.2, 0.5, 0.1, self.kernel_n2, tell='A2')
        n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        n_conv1_2 = self.gcn1_n_2(None, n_reduce2)
        n_conv1_2 = n_conv1_2 + n_conv1_3_shared # sum
        # n_conv1_2 = torch.cat((n_conv1_2, n_conv1_3_shared), -1) # concat
        n_conv2_2 = self.gcn2_n_2(n_conv1_2, n_reduce2)
        n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        # n_conv2_3_shared = self.gcn_n_shared1(n_conv1_2, n_reduce2)
        n_conv2_2 = n_conv2_2 + n_conv2_3_shared
        # n_conv3 = self.gcn3_n(n_conv2, n_reduce)
        # ----------------------------------
        # p_conv1_1_shared = self.gcn_p_shared(None, p_reduce)
        # p_conv1_2_shared = self.gcn_p_shared(None, p_reduce1)
        # p_conv1_3_shared = self.gcn_p_shared(None, p_reduce2)
        #
        # n_conv1_1_shared = self.gcn_n_shared(None, n_reduce)
        # n_conv1_2_shared = self.gcn_n_shared(None, n_reduce1)
        # n_conv1_3_shared = self.gcn_n_shared(None, n_reduce2)
        # -----------------------------------
        # p_conv = p_conv2 + p_conv2_1 + p_conv2_2
        # n_conv = n_conv2 + n_conv2_1 + n_conv2_2
        ##################################

        # conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat = torch.cat([p_conv2, n_conv2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat1 = torch.cat([p_conv2_1, n_conv2_1], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        conv_concat2 = torch.cat([p_conv2_2, n_conv2_2], -1).reshape([-1, 2 * self.in_dim * self.in_dim])
        output = self.lin2(self.lin1(conv_concat))
        output1 = self.lin2_1(self.lin1_1(conv_concat1))
        output2 = self.lin2_2(self.lin1_2(conv_concat2))
        # output = torch.softmax(output, dim=1)

        # F loss
        simi_loss1 = self.SimiLoss(self.kernel_p, self.kernel_p1)
        simi_loss2 = self.SimiLoss(self.kernel_p, self.kernel_p2)
        simi_loss3 = self.SimiLoss(self.kernel_p1, self.kernel_p2)
        simi_loss4 = self.SimiLoss(self.kernel_n, self.kernel_n1)
        simi_loss5 = self.SimiLoss(self.kernel_n, self.kernel_n2)
        simi_loss6 = self.SimiLoss(self.kernel_n1, self.kernel_n2)
        # simi_loss = squ_p1.sum() + squ_p2.sum() + squ_p3.sum() + squ_n1.sum() + squ_n2.sum() + squ_n3.sum()
        simiLoss = 0.1 * (simi_loss6 + simi_loss4 + simi_loss3 + simi_loss2 + simi_loss1 + simi_loss5)
        # 0.2 674 0.1 683 0.3 669 0.4 668 0.5 660 0.05 674 0.08 673 0.15 673

        score, score1, score2, score_, score_1, score_2, l1, l2, l3, l4, l5, l6 = self.load_s_c(self.kernel_p, self.kernel_p1, self.kernel_p2,
                                                                        self.kernel_n, self.kernel_n1, self.kernel_n2)
        # score, score1, score2, score_, score_1, score_2 = self.SimiLoss3(self.kernel_p, self.kernel_p1, self.kernel_p2,
        #                                                                  self.kernel_n, self.kernel_n1, self.kernel_n2)
        # 约束score max
        loss1 = -1 * torch.log(score + 1e-3)
        loss2 = -1 * torch.log(score1 + 1e-3)
        loss3 = -1 * torch.log(score2 + 1e-3)
        loss4 = -1 * torch.log(score_ + 1e-3)
        loss5 = -1 * torch.log(score_1 + 1e-3)
        loss6 = -1 * torch.log(score_2 + 1e-3)
        l = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        ll = l1 + l2 + l3 + l4 + l5 + l6
        # 625

        # score = self.load_s_c(self.kernel_p)
        # score1 = self.load_s_c(self.kernel_p1)
        # score2 = self.load_s_c(self.kernel_p2)
        SimiLoss1 = self.SimiLoss2(score, score1)
        SimiLoss2 = self.SimiLoss2(score, score2)
        SimiLoss3 = self.SimiLoss2(score1, score2)
        # score = score.view(6,1)
        # SimiLoss1 = F.cross_entropy(score.view(6, 1), score1.view(6, 1))

        # score_ = self.load_s_c(self.kernel_n)
        # score_1 = self.load_s_c(self.kernel_n1)
        # score_2 = self.load_s_c(self.kernel_n2)
        SimiLoss4 = self.SimiLoss2(score_, score_1)
        SimiLoss5 = self.SimiLoss2(score_, score_2)
        SimiLoss6 = self.SimiLoss2(score_1, score_2)
        simiLoss2 = 0 * (SimiLoss1 + SimiLoss6 + SimiLoss5 + SimiLoss4 + SimiLoss2 + SimiLoss3)
        simiLoss = simiLoss + l.sum() * 0.01 + ll.sum() * 0.0
        # simiLoss F相似性约束 0.1     l 子网络max    ll 子网络离散化

        # score max 0.01 683 0.001 677
        # 0.01l 0.1ll 663    0.1simiLoss 0.01l 0.1ll 662        0.01 0.01l 666           0.001 0.01 660     0.01 0.1 669
        # 0.01 1 656     0.01 0.5 655
        loss = torch.sum(torch.tensor(self.losses))+simiLoss
        self.losses.clear()
        loss1 = torch.sum(torch.tensor(self.losses1))
        self.losses1.clear()
        loss2 = torch.sum(torch.tensor(self.losses2))
        self.losses2.clear()
        return output, output1, output2, loss, loss1, loss2

    def SimiLoss(self, F1, F2):
        f1 = torch.nn.functional.relu(F1)
        f2 = torch.nn.functional.relu(F2).T
        O = torch.matmul(f1, f2)
        # O = O.trace()
        O0 = O.diagonal()
        O1 = F.softmax(O0)
        O2 = torch.log(O1).sum()
        # U = F.relu(O)
        # U1 = U.trace()
        # T = U.sum()
        # simi_loss1 = -torch.log(U1)
        simi_loss1 = -O2

        ####
        # simi_loss1 = 0
        # f1 = torch.nn.functional.relu(F1).T
        # f2 = torch.nn.functional.relu(F2).T
        # for i in range(8):
        #     L = nn.CrossEntropyLoss()
        #     l = L(f1[i], f2[i])
        #     simi_loss1 += l
        ####
        return simi_loss1
    def SimiLoss2(self, S1, S2):
        # CE loss
        # s1 = S1.unsqueeze(0)
        # s2 = S2.unsqueeze(0).T
        # O = torch.matmul(s1, s2)
        # # O = O.trace()
        # O1 = F.softmax(O)
        # O2 = torch.log(O1)
        # # U = F.relu(O)
        # # U1 = U.trace()
        # # T = U.sum()
        # # simi_loss1 = -torch.log(U1)
        # simi_loss2 = -O2

        # MSE loss
        # s1 = S1.unsqueeze(0)
        # s2 = S2.unsqueeze(0)
        # simi_loss2 = (abs(s1 - s2)).sum()

        s1 = S1.unsqueeze(1)
        s2 = S2.unsqueeze(1)
        # s1 = torch.log(S1.unsqueeze(1) + 0.0001)
        # s2 = torch.log(S2.unsqueeze(1) + 0.0001)
        simi_loss2 = self.mseLoss(s1, s2)
        return simi_loss2

    def load_s_c(self, F, F1, F2, F3, F4, F5):
        F = torch.nn.functional.relu(F).T
        F1 = torch.nn.functional.relu(F1).T
        F2 = torch.nn.functional.relu(F2).T
        F3 = torch.nn.functional.relu(F3).T
        F4 = torch.nn.functional.relu(F4).T
        F5 = torch.nn.functional.relu(F5).T
        s = F # 5 * 116
        s1 = F1 # 5 * 116
        s2 = F2 # 5 * 116
        s3 = F3 # 5 * 116
        s4 = F4 # 5 * 116
        s5 = F5 # 5 * 116
        s_MAX_INDEX = torch.argmax(s, dim=0)
        s_MAX_INDEX1 = torch.argmax(s1, dim=0)
        s_MAX_INDEX2 = torch.argmax(s2, dim=0)
        s_MAX_INDEX3 = torch.argmax(s3, dim=0)
        s_MAX_INDEX4 = torch.argmax(s4, dim=0)
        s_MAX_INDEX5 = torch.argmax(s5, dim=0)
        # print(s_MAX_INDEX)
        # ss = np.zeros((5, 116))
        ss = torch.zeros((self.in_dim, dim)).to('cuda:0')
        ss1 = torch.zeros((self.in_dim, dim)).to('cuda:0')
        ss2 = torch.zeros((self.in_dim, dim)).to('cuda:0')
        ss3 = torch.zeros((self.in_dim, dim)).to('cuda:0')
        ss4 = torch.zeros((self.in_dim, dim)).to('cuda:0')
        ss5 = torch.zeros((self.in_dim, dim)).to('cuda:0')
        for ii in range(dim):
            ss[s_MAX_INDEX[ii]][ii] = s[s_MAX_INDEX[ii]][ii]
            ss1[s_MAX_INDEX1[ii]][ii] = s1[s_MAX_INDEX1[ii]][ii]
            ss2[s_MAX_INDEX2[ii]][ii] = s2[s_MAX_INDEX2[ii]][ii]
            ss3[s_MAX_INDEX3[ii]][ii] = s3[s_MAX_INDEX3[ii]][ii]
            ss4[s_MAX_INDEX4[ii]][ii] = s4[s_MAX_INDEX4[ii]][ii]
            ss5[s_MAX_INDEX5[ii]][ii] = s5[s_MAX_INDEX5[ii]][ii]
        # s = ss
        # s1 = ss1
        # s2 = ss2
        # s3 = ss3
        # s4 = ss4
        # s5 = ss5
        R_sum = torch.sum(self.R, dim=1)
        scores = torch.zeros(self.nums).to('cuda:0')
        scores_ = torch.zeros(self.nums).to('cuda:0')
        scores1 = torch.zeros(self.nums).to('cuda:0')
        scores1_ = torch.zeros(self.nums).to('cuda:0')
        scores2 = torch.zeros(self.nums).to('cuda:0')
        scores2_ = torch.zeros(self.nums).to('cuda:0')
        scores3 = torch.zeros(self.nums).to('cuda:0')
        scores3_ = torch.zeros(self.nums).to('cuda:0')
        scores4 = torch.zeros(self.nums).to('cuda:0')
        scores4_ = torch.zeros(self.nums).to('cuda:0')
        scores5 = torch.zeros(self.nums).to('cuda:0')
        scores5_ = torch.zeros(self.nums).to('cuda:0')
        for ri in range(self.nums):
            tmp_sum = 0
            tmp_sum1 = 0
            tmp_sum2 = 0
            tmp_sum3 = 0
            tmp_sum4 = 0
            tmp_sum5 = 0

            tmp_sum_ = 0
            tmp_sum_1 = 0
            tmp_sum_2 = 0
            tmp_sum_3 = 0
            tmp_sum_4 = 0
            tmp_sum_5 = 0
            temp = self.ij[ri]
            for ij in temp:
                i = ij[0]
                j = ij[1]

                t = (s[:, i] * s[:, j])
                tmp_sum_ += t.sum()
                t1 = s1[:, i] * s1[:, j]
                tmp_sum_1 += t1.sum()
                t2 = s2[:, i] * s2[:, j]
                tmp_sum_2 += t2.sum()
                t3 = s3[:, i] * s3[:, j]
                tmp_sum_3 += t3.sum()
                t4 = s4[:, i] * s4[:, j]
                tmp_sum_4 += t4.sum()
                t5 = s5[:, i] * s5[:, j]
                tmp_sum_5 += t5.sum()

                t = torch.matmul(ss[:, i].unsqueeze(1), ss[:, j].unsqueeze(1).T)
                t = t - torch.diag_embed(torch.diag(t))
                tmp_sum += t.sum()
                t1 = torch.matmul(ss1[:, i].unsqueeze(1), ss1[:, j].unsqueeze(1).T)
                t1 = t1 - torch.diag_embed(torch.diag(t1))
                tmp_sum1 += t1.sum()
                t2 = torch.matmul(ss2[:, i].unsqueeze(1), ss2[:, j].unsqueeze(1).T)
                t2 = t2 - torch.diag_embed(torch.diag(t2))
                tmp_sum2 += t2.sum()
                t3 = torch.matmul(ss3[:, i].unsqueeze(1), ss3[:, j].unsqueeze(1).T)
                t3 = t3 - torch.diag_embed(torch.diag(t3))
                tmp_sum3 += t3.sum()
                t4 = torch.matmul(ss4[:, i].unsqueeze(1), ss4[:, j].unsqueeze(1).T)
                t4 = t4 - torch.diag_embed(torch.diag(t4))
                tmp_sum4 += t4.sum()
                t5 = torch.matmul(ss5[:, i].unsqueeze(1), ss5[:, j].unsqueeze(1).T)
                t5 = t5 - torch.diag_embed(torch.diag(t5))
                tmp_sum5 += t5.sum()

            scores_[ri] = tmp_sum_
            scores1_[ri] = tmp_sum_1
            scores2_[ri] = tmp_sum_2
            scores3_[ri] = tmp_sum_3
            scores4_[ri] = tmp_sum_4
            scores5_[ri] = tmp_sum_5

            scores[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum
            scores1[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum1
            scores2[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum2
            scores3[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum3
            scores4[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum4
            scores5[ri] = (2 / (R_sum[ri] ** 2)) * tmp_sum5
        return scores, scores1, scores2, scores3, scores4, scores5, scores_, scores1_, scores2_, scores3_, scores4_, scores5_


def evaluate(dataset, model, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()
    avg_loss = 0.0
    preds = []
    preds1 = []
    preds2 = []
    labels = []
    ypreds = []
    ypreds1 = []
    ypreds2 = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
            adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            labels.append(data['label'].long().numpy())
            ypred, ypred1, ypred2, losses, losses1, losses2 = model(adj, adj1, adj2)
            loss = F.cross_entropy(ypred, label, size_average=True)
            loss1 = F.cross_entropy(ypred1, label, size_average=True)
            loss2 = F.cross_entropy(ypred2, label, size_average=True)
            loss += losses
            loss1 += losses1
            loss2 += losses2
            for i in ypred:
                ypreds.append(np.array(i.cpu()))
            for i in ypred1:
                ypreds1.append(np.array(i.cpu()))
            for i in ypred2:
                ypreds2.append(np.array(i.cpu()))

            avg_loss += loss
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            _, indices = torch.max(ypred1, 1)
            preds1.append(indices.cpu().data.numpy())
            _, indices = torch.max(ypred2, 1)
            preds2.append(indices.cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * 32 > max_num_examples:
                    break
    avg_loss /= batch_idx + 1

    ypres_all = np.array(ypreds)
    ypres_all1 = np.array(ypreds1)
    ypres_all2 = np.array(ypreds2)

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    preds1 = np.hstack(preds1)
    preds2 = np.hstack(preds2)

    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
    result = {'prec': metrics.precision_score(labels, preds),
              'recall': metrics.recall_score(labels, preds),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds)}
    auc = metrics.roc_auc_score(labels, preds1, average='macro', sample_weight=None)
    result1 = {'prec': metrics.precision_score(labels, preds1),
              'recall': metrics.recall_score(labels, preds1),
              'acc': metrics.accuracy_score(labels, preds1),
              'F1': metrics.f1_score(labels, preds1, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds1)}
    auc = metrics.roc_auc_score(labels, preds2, average='macro', sample_weight=None)
    result2 = {'prec': metrics.precision_score(labels, preds2),
              'recall': metrics.recall_score(labels, preds2),
              'acc': metrics.accuracy_score(labels, preds2),
              'F1': metrics.f1_score(labels, preds2, average="macro"),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds2)}
    xx = preds
    yy = labels

    return avg_loss, result, result1, result2, ypres_all, ypres_all1, ypres_all2


def evaluate_all(dataset, preds):
    labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            labels.append(data['label'].long().numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, preds, average='macro', sample_weight=None)
    result = {
        # 'prec': metrics.precision_score(labels, preds),
        #       'recall': metrics.recall_score(labels, preds),
              'acc': metrics.accuracy_score(labels, preds),
              # 'F1': metrics.f1_score(labels, preds, average="macro"),
              # 'auc': auc,
              # 'matrix': confusion_matrix(labels, preds)
    }
    xx = preds
    yy = labels

    return result, preds


result = [0, 0, 0, 0, 0]
ii = 0

# def train_original(test_data_loaders, dataset, student, val_dataset=None, test_dataset=None,
#           device='cpu', phi=None, early_epoch=200, middle_epoch=300, late_epoch=500, supernode=8, fold=0):
#
#     optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.0001,
#                                   weight_decay=0.001)
#     cosinLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20, eta_min=0.000001)
#     for name in student.state_dict():
#         print(name)
#     iter = 0
#     best_val_acc = 0.0
#     #early_stopping = EarlyStopping(patience=40, verbose=True)
#     bestVal = []
#     best = 0
#     global ii
#     ii += 1
#
#     for epoch in range(1, late_epoch+1):
#         begin_time = time.time()
#         avg_loss = 0.0
#         student.train()
#         print(epoch)
#         print("train_for_original")
#         # pbar = len(dataset)
#         for idx, data in enumerate(dataset):
#
#             if epoch < 0:
#                 for k, v in student.named_parameters():
#                     if k != 'gcn1_p.kernel' and k != 'gcn2_p.kernel' and k != 'gcn3_p.kernel' and k != 'gcn1_n.kernel' and k != 'gcn2_n.kernel' and k != 'gcn3_n.kernel':
#                         v.requires_grad = False  # 固定参数
#                 time1 = time.time()
#                 student.zero_grad()
#                 adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
#                 label = Variable(data['label'].long()).to(device)
#                 pred, losses = student(adj)
#                 loss = F.cross_entropy(pred, label, size_average=True)
#                 loss += losses
#                 loss.backward()
#                 time3 = time.time()
#                 nn.utils.clip_grad_norm_(student.parameters(), 2.0)
#                 optimizer1.step()
#                 iter += 1
#                 avg_loss += loss
#             else:
#                 for k, v in student.named_parameters():
#                     v.requires_grad = True
#                 time1 = time.time()
#                 student.zero_grad()
#
#
#                 adj_train = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
#                 adj1_train = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
#                 adj2_train = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
#
#                 label = Variable(data['label'].long(), requires_grad=False).to(device)
#
#                 pred, pred1, pred2, losses, losses1, losses2 = student(adj_train, adj1_train, adj2_train)
#
#                 loss = F.cross_entropy(pred, label, size_average=True)
#                 loss1 = F.cross_entropy(pred1, label, size_average=True)
#                 loss2 = F.cross_entropy(pred2, label, size_average=True)
#                 loss += losses
#                 loss1 += losses1
#                 loss2 += losses2
#
#                 loss = loss + loss1 + loss2
#
#                 loss.backward()
#
#                 time3 = time.time()
#                 nn.utils.clip_grad_norm_(student.parameters(), 2.0)
#                 optimizer2.step()
#
#                 iter += 1
#                 avg_loss += loss
#
#         avg_loss /= idx + 1
#         # print(avg_loss)
#         eval_time = time.time()
#         if val_dataset is not None:
#             _, train_result, train_result1, train_result2, _, _, _ = evaluate(dataset, student, name='Train', device=device)
#             val_loss, val_result, val_result1, val_result2, _, _, _ = evaluate(val_dataset, student, name='Validation', device=device)
#             _, _, _, _, pre, pre1, pre2 = evaluate(test_data_loaders, student, name='Test', device=device)
#             pre_all = pre + pre1 + pre2
#
#             pre_all = pre_all.tolist()
#             temp = np.array(pre_all)
#             res = np.argmax(temp, axis=1)
#
#             pre_all = np.array(res)
#
#             test_result, _ = evaluate_all(test_data_loaders, pre_all)
#
#             print('train1', train_result)
#             print('val1', val_result)
#             print('train2', train_result1)
#             print('val2', val_result1)
#             print('train3', train_result2)
#             print('val3', val_result2)
#
#     for k, v in student.named_parameters():
#         v.requires_grad = False
#
#     uncertainty_scores = []
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(dataset):
#             preds = []
#             preds1 = []
#             preds2 = []
#             adj = Variable(batch_data['adj'].to(torch.float32), requires_grad=False).to(device)
#             adj1 = Variable(batch_data['adj1'].to(torch.float32), requires_grad=False).to(device)
#             adj2 = Variable(batch_data['adj2'].to(torch.float32), requires_grad=False).to(device)
#             pred, pred1, pred2, losses, losses1, losses2 = student(adj, adj1, adj2)
#             for i in pred:
#                 preds.append(np.array(i.cpu()))
#             for i in pred1:
#                 preds1.append(np.array(i.cpu()))
#             for i in pred2:
#                 preds2.append(np.array(i.cpu()))
#             pres_all = np.array(preds)
#             pres_all1 = np.array(preds1)
#             pres_all2 = np.array(preds2)
#             pre_all = pres_all + pres_all1 + pres_all2
#             # 计算不确定性
#             batch_uncertainty_scores = uncertainty(pre_all)
#             for idx, uncertainty_score in enumerate(batch_uncertainty_scores):
#                 uncertainty_scores.append((uncertainty_score.item(), batch_idx * args.batch_size + idx))
#         uncertainty_scores.sort(reverse=True)
#         # 选择不确定性最大的 10% 数据
#         top_10_percent = int(len(uncertainty_scores) * 0.1)
#         selected_indices = [idx for _, idx in uncertainty_scores[:top_10_percent]]
#
#     print(bestVal)
#     print(best)
#     # model.load_state_dict(torch.load('./GroupINN_model/checkpoint' + str(phi) + '_' + str(ii) + '.pt'))
#     return student, selected_indices

#
# def train_original(test_data_loaders, dataset, student, val_dataset=None, test_dataset=None,
#                    device='cpu', phi=None, early_epoch=200, middle_epoch=300, late_epoch=500,
#                    supernode=8, fold=0):
#     """
#     Implement the progressive uncertainty-based outlier screening with three-phase training
#     as described in the paper section "Progressive Uncertainty-Based Outlier Screening"
#     """
#     # Initialize optimizer and scheduler
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()),
#                                  lr=0.0001, weight_decay=0.001)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)
#
#     # Storage for uncertainty scores across three phases
#     uncertainty_early = []  # Early phase uncertainty scores
#     uncertainty_middle = []  # Middle phase uncertainty scores
#     uncertainty_late = []  # Late phase uncertainty scores
#
#     best_val_acc = 0.0
#     best_model_state = None
#
#     # Training loop across all epochs
#     for epoch in range(1, late_epoch + 1):
#         student.train()
#         avg_loss = 0.0
#
#         # # Phase determination
#         # current_phase = None
#         # if epoch <= early_epoch:
#         #     current_phase = "early"
#         # elif epoch <= middle_epoch:
#         #     current_phase = "middle"
#         # else:
#         #     current_phase = "late"
#
#         # Batch training
#         for idx, data in enumerate(dataset):
#             student.zero_grad()
#
#             # Prepare batch data
#             adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
#             adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
#             adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
#             label = Variable(data['label'].long()).to(device)
#
#
#             pred, pred1, pred2, losses, losses1, losses2 = student(adj, adj1, adj2)
#
#             # Combined loss (Equation 15)
#             loss = F.cross_entropy(pred, label) + \
#                    F.cross_entropy(pred1, label) + \
#                    F.cross_entropy(pred2, label) + \
#                    losses + losses1 + losses2
#
#             # Backward pass
#             loss.backward()
#             nn.utils.clip_grad_norm_(student.parameters(), 2.0)
#             optimizer.step()
#             avg_loss += loss.item()
#
#         scheduler.step()
#
#         # Calculate uncertainty at phase transition points (Equations 11-13)
#         if epoch == early_epoch or epoch == middle_epoch or epoch == late_epoch:
#             current_uncertainties = []
#             student.eval()
#
#             with torch.no_grad():
#                 for batch_data in dataset:
#                     adj = Variable(batch_data['adj'].to(torch.float32), requires_grad=False).to(device)
#                     adj1 = Variable(batch_data['adj1'].to(torch.float32), requires_grad=False).to(device)
#                     adj2 = Variable(batch_data['adj2'].to(torch.float32), requires_grad=False).to(device)
#
#                     # Get predictions from all views
#                     pred, pred1, pred2, _, _, _ = student(adj, adj1, adj2)
#
#                     # Combine predictions (softmax probabilities)
#                     combined_pred = (F.softmax(pred, dim=1) +
#                                      F.softmax(pred1, dim=1) +
#                                      F.softmax(pred2, dim=1)) / 3
#
#                     # Calculate entropy-based uncertainty (Equation 12)
#                     entropy = -torch.sum(combined_pred * torch.log(combined_pred + 1e-10), dim=1)
#                     current_uncertainties.extend(entropy.cpu().numpy())
#
#             # Store uncertainties by phase
#             if epoch == early_epoch:
#                 uncertainty_early = current_uncertainties
#             elif epoch == middle_epoch:
#                 uncertainty_middle = current_uncertainties
#             else:
#                 uncertainty_late = current_uncertainties
#
#         # Validation and evaluation
#         if val_dataset is not None:
#             student.eval()
#             with torch.no_grad():
#                 # Training set evaluation
#                 _, train_result, train_result1, train_result2, _, _, _ = evaluate(
#                     dataset, student, name='Train', device=device)
#
#                 val_loss, val_result, val_result1, val_result2, _, _, _ = evaluate(
#                     val_dataset, student, name='Validation', device=device)
#
#
#                 _, _, _, _, pre, pre1, pre2 = evaluate(
#                     test_data_loaders, student, name='Test', device=device)
#
#
#                 if val_result['accuracy'] > best_val_acc:
#                     best_val_acc = val_result['accuracy']
#                     best_model_state = student.state_dict()
#
#                 print(f'Epoch {epoch}:')
#                 print(f'Train Accuracy: {train_result["accuracy"]:.4f} | Val Accuracy: {val_result["accuracy"]:.4f}')
#
#
#     weights = {'early': 0.2, 'middle': 0.3, 'late': 0.5}
#     #weights = {'early': 0.1, 'middle': 0.2, 'late': 0.7}
#     #weights = {'early': 0.2, 'middle': 0.4, 'late': 0.4}
#     final_scores = (
#             weights['early'] * np.array(uncertainty_early) +
#             weights['middle'] * np.array(uncertainty_middle) +
#             weights['late'] * np.array(uncertainty_late)
#     )
#
#     sorted_indices = np.argsort(final_scores)[::-1]
#     top_k = int(len(sorted_indices) * 0.10)
#     selected_indices = sorted_indices[:top_k]
#
#
#     if best_model_state is not None:
#         student.load_state_dict(best_model_state)
#
#     return student, selected_indices





def train_original(test_data_loaders, dataset, student, val_dataset=None, test_dataset=None,
          device='cpu', phi=None, early_epoch=200, middle_epoch=300, late_epoch=500,
          supernode=8, fold=0, wearly=0.3, wmiddle=0.3, wlate=0.4, top_ratio=0.1):

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.0001,
                                  weight_decay=0.001)
    cosinLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20, eta_min=0.000001)

    iter = 0
    best_val_acc = 0.0
    bestVal = []
    best = 0
    global ii
    ii += 1

    # 保存各阶段的不确定性
    uncertainty_dict = {"early": None, "middle": None, "late": None}

    for epoch in range(1, late_epoch+1):
        student.train()
        avg_loss = 0.0
        print(f"Epoch {epoch} / {late_epoch} (train_for_original)")

        for idx, data in enumerate(dataset):
            student.zero_grad()
            for k, v in student.named_parameters():
                v.requires_grad = True

            adj_train = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            adj1_train = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
            adj2_train = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long(), requires_grad=False).to(device)

            pred, pred1, pred2, losses, losses1, losses2 = student(adj_train, adj1_train, adj2_train)

            loss = F.cross_entropy(pred, label, reduction="mean")
            loss1 = F.cross_entropy(pred1, label, reduction="mean")
            loss2 = F.cross_entropy(pred2, label, reduction="mean")
            loss = loss + loss1 + loss2 + losses + losses1 + losses2

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer2.step()

            iter += 1
            avg_loss += loss.item()

        avg_loss /= (idx + 1)

        # ---- 在 early / middle / late 阶段保存不确定性 ----
        if epoch in [early_epoch, middle_epoch, late_epoch]:
            phase = "early" if epoch == early_epoch else ("middle" if epoch == middle_epoch else "late")
            print(f"===> Collecting {phase} uncertainty at epoch {epoch}")
            uncertainty_scores = []
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dataset):
                    preds, preds1, preds2 = [], [], []
                    adj = Variable(batch_data['adj'].to(torch.float32), requires_grad=False).to(device)
                    adj1 = Variable(batch_data['adj1'].to(torch.float32), requires_grad=False).to(device)
                    adj2 = Variable(batch_data['adj2'].to(torch.float32), requires_grad=False).to(device)
                    pred, pred1, pred2, _, _, _ = student(adj, adj1, adj2)

                    preds.extend([np.array(i.cpu()) for i in pred])
                    preds1.extend([np.array(i.cpu()) for i in pred1])
                    preds2.extend([np.array(i.cpu()) for i in pred2])

                    pre_all = np.array(preds) + np.array(preds1) + np.array(preds2)
                    batch_uncertainty_scores = uncertainty(pre_all)

                    for idx_u, uncertainty_score in enumerate(batch_uncertainty_scores):
                        uncertainty_scores.append((uncertainty_score.item(),
                                                   batch_idx * args.batch_size + idx_u))
            # 保存当前阶段不确定性
            uncertainty_dict[phase] = uncertainty_scores

    # === 训练结束后，融合早中晚三个阶段的不确定性 ===
    print("Combining uncertainties from early, middle, late phases...")
    final_uncertainty = []

    for i in range(len(uncertainty_dict["late"])):
        u_early = uncertainty_dict["early"][i][0]
        u_middle = uncertainty_dict["middle"][i][0]
        u_late = uncertainty_dict["late"][i][0]
        idx_sample = uncertainty_dict["late"][i][1]  # 样本 index

        u_final = wearly * u_early + wmiddle * u_middle + wlate * u_late
        final_uncertainty.append((u_final, idx_sample))

    final_uncertainty.sort(reverse=True)
    top_k = int(len(final_uncertainty) * top_ratio)
    selected_indices = [idx for _, idx in final_uncertainty[:top_k]]

    return student, selected_indices





def train(raw_data, test_data_loaders, dataset, student, val_dataset=None, test_dataset=None,
          device='cpu', phi=None, e=5, supernode=8, fold=0):

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=0.0001,
                                  weight_decay=0.001)
    cosinLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20, eta_min=0.000001)
    for name in student.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0.0
    #early_stopping = EarlyStopping(patience=40, verbose=True)
    bestVal = []
    best = 0
    global ii
    ii += 1

    for epoch in range(1, e+1):
        begin_time = time.time()
        avg_loss = 0.0
        student.train()
        print(epoch)
        print("train for semi")
        pbar = len(dataset)
        for idx, data in enumerate(dataset):
            # print(len(dataset))
            # print(idx)
            # print(1/0)
            if epoch < 0:
                for k, v in student.named_parameters():
                    if k != 'gcn1_p.kernel' and k != 'gcn2_p.kernel' and k != 'gcn3_p.kernel' and k != 'gcn1_n.kernel' and k != 'gcn2_n.kernel' and k != 'gcn3_n.kernel':
                        v.requires_grad = False  # 固定参数
                time1 = time.time()
                student.zero_grad()
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                pred, losses = student(adj)
                loss = F.cross_entropy(pred, label, size_average=True)
                loss += losses
                loss.backward()
                time3 = time.time()
                nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                optimizer1.step()
                iter += 1
                avg_loss += loss
            else:
                for k, v in student.named_parameters():
                    v.requires_grad = True
                time1 = time.time()
                student.zero_grad()


                #不加噪声的原始数据
                adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
                adj1 = Variable(data['adj1'].to(torch.float32), requires_grad=False).to(device)
                adj2 = Variable(data['adj2'].to(torch.float32), requires_grad=False).to(device)
                label = Variable(data['label'].long()).to(device)
                pred, pred1, pred2, losses, losses1, losses2 = student(adj, adj1, adj2)


                loss = F.cross_entropy(pred, label, size_average=True)
                loss1 = F.cross_entropy(pred1, label, size_average=True)
                loss2 = F.cross_entropy(pred2, label, size_average=True)
                loss += losses
                loss1 += losses1
                loss2 += losses2


                loss = loss + loss1 + loss2


                loss.backward()

                time3 = time.time()
                nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                optimizer2.step()
                # ema_update(teacher=teacher, student=student, ema_decay=0.99, cur_step=idx + pbar * (epoch - 1))
                # cosinLR2.step()
                iter += 1
                avg_loss += loss

        avg_loss /= idx + 1
        # print(avg_loss)
        eval_time = time.time()
        if val_dataset is not None:
            _, train_result, train_result1, train_result2, _, _, _ = evaluate(dataset, student, name='Train', device=device)
            val_loss, val_result, val_result1, val_result2, _, _, _ = evaluate(val_dataset, student, name='Validation', device=device)
            _, _, _, _, pre, pre1, pre2 = evaluate(test_data_loaders, student, name='Test', device=device)
            pre_all = pre + pre1 + pre2

            pre_all = pre_all.tolist()
            temp = np.array(pre_all)
            res = np.argmax(temp, axis=1)

            pre_all = np.array(res)

            test_result, _ = evaluate_all(test_data_loaders, pre_all)
            if fold == 0:
                # prec_1.append(test_result["prec"])
                # recall_1.append(test_result["recall"])
                acc_1.append(test_result["acc"])
                # F1_1.append(test_result["F1"])
                # auc_1.append(test_result["auc"])

            elif fold == 1:
                # prec_2.append(test_result["prec"])
                # recall_2.append(test_result["recall"])
                acc_2.append(test_result["acc"])
                # F1_2.append(test_result["F1"])
                # auc_2.append(test_result["auc"])

            elif fold == 2:
                # prec_3.append(test_result["prec"])
                # recall_3.append(test_result["recall"])
                acc_3.append(test_result["acc"])
                # F1_3.append(test_result["F1"])
                # auc_3.append(test_result["auc"])

            elif fold == 3:
                # prec_4.append(test_result["prec"])
                # recall_4.append(test_result["recall"])
                acc_4.append(test_result["acc"])
                # F1_4.append(test_result["F1"])
                # auc_4.append(test_result["auc"])

            elif fold == 4:
                # prec_5.append(test_result["prec"])
                # recall_5.append(test_result["recall"])
                acc_5.append(test_result["acc"])
                # F1_5.append(test_result["F1"])
                # auc_5.append(test_result["auc"])

            # print('',)
            print('train1', train_result)
            print('val1', val_result)
            print('train2', train_result1)
            print('val2', val_result1)
            print('train3', train_result2)
            print('val3', val_result2)

    print(bestVal)
    print(best)
    # model.load_state_dict(torch.load('./GroupINN_model/checkpoint' + str(phi) + '_' + str(ii) + '.pt'))
    return student
###########################################################################################
###########################################################################################

# 主函数


def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent


def main():

    # 设置种子
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 导入数据
    _, _, raw_data, labels = load_data(args.data_path, args.label_path)  # raw_data [871 116 ?]  labels [871]
    ind = 0
    for i in range(871):
        if labels[i] == 0:
            ind += 1
    print(ind)
    # 划分时间窗

    print('finished')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device)

    print('process the A')
    adj = cal_pcc(raw_data, args.phi)
    print(args.phi, 'Done')
    adj1 = cal_pcc(raw_data, args.phi1)
    print(args.phi1, 'Done')
    adj2 = cal_pcc(raw_data, args.phi2)
    print(args.phi2, 'Done')

    raw_data_number = np.arange(871)
    # train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(raw_data, args.phi, args.phi1, args.phi2, adj, adj1, adj2, labels)
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(raw_data_number, adj, adj1, adj2, labels)
    result = []
    result1 = []
    result2 = []
    pres = []
    jj=1
    selected_indices_list = []
    for i in range(len(train_data_loaders)):
        student = model_gnn(8, 8, 2)

        jj+=1
        student.to(device)

        model, selected_indices = train_original(test_data_loaders[i], train_data_loaders[i], student, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, early_epoch=args.epochs_early, middle_epoch=args.epochs_middle, late_epoch=args.epochs_late,
                                                 supernode=args.super_nodes, fold=i, wearly=0.2, wmiddle=0.3, wlate=0.5, top_ratio=args.ratio_outlier)


        selected_indices_list.append(selected_indices)


        _, test_result, test_result1, test_result2, pre, pre1, pre2 = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pre_all = pre + pre1 + pre2
        pres.append(pre_all)
        print('test1', test_result)
        print('test2', test_result1)
        print('test3', test_result2)
        result.append(test_result)
        result1.append(test_result1)
        result2.append(test_result2)
        del model
        del test_result
        del test_result1
        del test_result2
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val_semi(selected_indices_list, raw_data_number, adj, adj1, adj2, labels)
    result = []
    result1 = []
    result2 = []
    pres = []
    jj=1
    selected_indices_list = []
    for i in range(len(train_data_loaders)):
        student = model_gnn(8, 8, 2)

        jj+=1
        student.to(device)

        model = train(raw_data, test_data_loaders[i], train_data_loaders[i], student, val_dataset=valid_data_loaders[i],
                      test_dataset=test_data_loaders[i], device=device, phi=0.6, e=args.epochs, supernode=args.super_nodes, fold=i)

        _, test_result, test_result1, test_result2, pre, pre1, pre2 = evaluate(test_data_loaders[i], model, name='Test', device=device)
        pre_all = pre + pre1 + pre2
        pres.append(pre_all)
        print('test1', test_result)
        print('test2', test_result1)
        print('test3', test_result2)
        result.append(test_result)
        result1.append(test_result1)
        result2.append(test_result2)
        del model
        del test_result
        del test_result1
        del test_result2
    print(result)
    print(result1)
    print(result2)
    print('------------------------------------')

    for i in range(len(train_data_loaders)):
        pres[i] = pres[i].tolist()
        temp = np.array(pres[i])
        res = np.argmax(temp, axis=1)

        pres[i] = np.array(res)
        test_result, _ = evaluate_all(test_data_loaders[i], pres[i])
        print(test_result)
        result.append(test_result)


if __name__ == "__main__":
    main()
