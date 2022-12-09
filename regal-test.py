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
    
    def __int__(self, input = 'data/arenas_combined_edges.txt', output = 'emb/arenas990-1.emb', attributes = None,
                attrvals = 2, dimensions = 128, k = 10, untillayer = 2, alpha = 0.01, gammastruc = 1, gammaattr = 1,
                numtop = 10, buckets = 2):
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


if __name__ == "__main__":
    input = 'data/arenas_combined_edges.txt'
    output = 'emb/arenas990-1.emb'
    attributes = None
    attrvals = 2
    dimensions = 128
    k = 10
    untillayer = 2
    alpha = 0.01
    gammastruc = 1
    gammaattr = 1
    numtop = 10
    buckets = 2
    arg = arg_attribute()

    # 查看数据集的名称
    dataset_name = arg.output.split("/")
    if len(dataset_name) == 1:
        dataset_name = dataset_name[-1].split(".")[0]
    else:
        dataset_name = dataset_name[-2]
    print("数据集名称", dataset_name)
    #
    true_alignments_fname = input.split("_")[0] + "_edges-mapping-permutation.txt"  # can be changed if desired
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
    
    if attributes is not None:
        attributes = np.load(attributes)  # 从文件加载属性向量
        print("属性向量读取成功！")
        print(attributes.shape)
        
    

