# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 21:17
# @Author  : CX
# @Email   : 2497068245@qq.com
# @File    : regal-test.py
# @Software: PyCharm

import numpy as np
import argparse  # 导入argparse包
# argparse 模块是 Python 内置的用于命令项选项与参数解析的模块，
# argparse 模块可以让人轻松编写用户友好的命令行接口，能够帮助程序员为模型定义参数。
import networkx as nx
import time
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix

import xnetmf
from config import *
from alignments import *


class arg_attribute:
    def __init__(self,
                 input='data/arenas_combined_edges.txt',
                 output='emb/arenas990-1.emb',
                 attributes=None,
                 attrvals=2,
                 dimensions=128,
                 k=10,
                 untillayer=2,
                 alpha=0.01,
                 gammastruc=1,
                 gammaattr=1,
                 numtop=10,
                 buckets=2):
        self.input = input
        self.output = output
        self.attributes = attributes
        self.attrvals = attrvals
        self.dimensions = dimensions
        self.k = k
        self.untillayer = untillayer
        self.alpha = alpha
        self.gammastruc = gammastruc
        self.gammaattr = gammaattr
        self.numtop = numtop
        self.buckets = buckets
        print("构造方法被调用")



def learn_representations(args):
    nx_graph = nx.read_edgelist(args.input, nodetype=int, comments="%")
    print("读入图")
    adj = nx.adjacency_matrix(nx_graph, nodelist=range(nx_graph.number_of_nodes()))
    print("得到邻接矩阵")

    graph = Graph(adj, node_attributes=args.attributes)
    max_layer = args.untillayer   # xNetMF层之前的计算，默认为2
    if args.untillayer == 0:
        max_layer = None
    alpha = args.alpha   # 折现因子，默认为0.01
    num_buckets = args.buckets   # degree binning)的对数基  BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    # 创建对象
    rep_method = RepMethod(max_layer=max_layer,        # xNetMF层之前的计算,			 	  默认为2
                           alpha=alpha,                # 折现因子,						  默认为0.01
                           k=args.k,                   # 控制要采样的地标(landmark)数量，	  默认值为10
                           num_buckets=num_buckets,    # 度（节点特征）装箱(degree binning)的对数基
                           normalize=True,             # 是否标准化
                           gammastruc=args.gammastruc, # 结构相似性权重，					  默认为1
                           gammaattr=args.gammaattr)   # 属性相似性权重， 					  默认为1
    if max_layer is None:
        max_layer = 1000
    # print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
    print("最大层=%d 折现因子=%f,进行表示学习" % (max_layer, alpha))
    # 学习表示
    representations = xnetmf.get_representations(graph, rep_method)
    np.save(args.output, representations)
    return representations


if __name__ == "__main__":
    print(np.version.version)
    os.chdir("E:\workplace\GithubDesktop\REGAL")
    print("当前工作目录:", os.getcwd())
    print(os.path.abspath(__file__))
    print("当前工作目录:", os.path.abspath(os.getcwd()))
    # 创建参数对象
    arg = arg_attribute()
    print(arg.input, arg.output)
    # 查看数据集的名称
    dataset_name = arg.output.split("/")
    if len(dataset_name) == 1:
        dataset_name = dataset_name[-1].split(".")[0]
    else:
        dataset_name = dataset_name[-2]
    print("数据集名称", dataset_name)
    # 真实标签文件目录
    true_alignments_fname = arg.input.split("_")[0] + "_edges-mapping-permutation.txt"  # can be changed if desired
    print("真实标签文件目录: ", true_alignments_fname)
    true_alignments = None
    if os.path.exists(true_alignments_fname):
        print("真实标签文件存在！")
        with open(true_alignments_fname, "rb") as true_alignments_file:
            try:
                true_alignments = pickle.load(true_alignments_file)
            except:
                true_alignments = pickle.load(true_alignments_file, encoding="latin1")
        print("真实标签文件读取成功！")
    else:
        print("真实标签文件不存在！")

    if arg.attributes is not None:
        arg.attributes = np.load(arg.attributes)  # 从文件加载属性向量
        print("属性向量读取成功！")
        print(arg.attributes.shape)
    else:
        print("属性向量不存在！")

    result = np.load("emb/arenas990-1.emb.npy")

    print("读取成功", result)
    # print("表示学习中...")
    # before_rep = time.time()
    # embed = learn_representations(arg)
    # print(embed)
    # after_rep = time.time()
    # print("表示学习用时 %f 秒" % (after_rep - before_rep))


