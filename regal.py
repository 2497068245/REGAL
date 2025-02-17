## 导入第三方库
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

## 调用内部文件
import xnetmf
from config import *
from alignments import *


# from .. import .. 不但可以拿到py文件(也叫模块)，也可以拿到模块中的类，或函数，可以直接使用函数名，类名。
# import .. 只能拿到py文件(也叫模块)，通过模块名去调用py文件中的类，或函数。比如 numpy.array


# argparse定义四个步骤：
# 1 导入argparse包 				——	mport argparse
# 2 创建一个命令行解析器对象		—— 创建 ArgumentParser() 对象
# 3 给解析器添加命令行参数		   ——  调用add_argument() 方法添加参数
# 4 解析命令行的参数 			  ——  使用 parse_args() 解析添加的参数
def parse_args():
    parser = argparse.ArgumentParser(description="运行 REGAL.")  # 2 创建一个命令行解析器对象
    # 3 给解析器添加命令行参数
    # 输入文件（所有边的列表）位置：默认是 data/arenas_combined_edges.txt
    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt',
                        help="一张Edgelist,可以包含两张以上图")

    # 输出文件位置，默认是：emb/arenas990-1.emb
    parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
                        help='嵌入路径Embeddings path')
    # 特征文件位置，默认为 None。文件中保存了节点属性的numpy矩阵，或要综合生成的属性数量的int。默认值为5合成。
    parser.add_argument('--attributes', nargs='?', default=None,
                        help='保存结点属性的numpy矩阵文件 或者 要生成的合成属性的数量(int), 默认是None,生成5个合成属性.')
    # 属性数量，默认为2。属性值的数量。仅在生成合成属性时使用
    parser.add_argument('--attrvals', type=int, default=2,
                        help='属性值的数量. 默认是2')
    # 维度，默认为128
    parser.add_argument('--dimensions', type=int, default=128,
                        help='维度. 默认是128')
    # 控制要采样的地标(landmark)数量，默认值为10
    parser.add_argument('--k', type=int, default=10,
                        help='要采样的landmark数量, 默认是10')
    # xNetMF层之前的计算，默认为2
    parser.add_argument('--untillayer', type=int, default=2,
                        help='xNetMF层之前的计算, 默认是2')
    # 其他层的折扣系数(Discount factor,也叫贴现因子、折现系数、折现参数、折现因子)，默认为0.01
    parser.add_argument('--alpha', type=float, default=0.01, help="下一层的折现因子(discount factor), 默认为0.01")
    # 结构相似性权重
    parser.add_argument('--gammastruc', type=float, default=1, help="结构相似性权重, 默认为1")
    # 属性相似性权重
    parser.add_argument('--gammaattr', type=float, default=1, help="属性相似度权重, 默认为1")
    parser.add_argument('--numtop', type=int, default=10,
                        help="使用kd树计算的top相似度的数量。默认为10,如果为0,则计算所有成对相似度")
    # 度（节点特征）装箱(degree binning)的对数基
    parser.add_argument('--buckets', type=float, default=2, help="度（节点特征）装箱(degree binning)的对数基, 默认为2")
    return parser.parse_args()  # 4 解析命令行的参数 ——使用 parse_args() 解析添加的参数


def main(args):
    # 提取数据集名称
    dataset_name = args.output.split("/")
    if len(dataset_name) == 1:
        dataset_name = dataset_name[-1].split(".")[0]
    else:
        dataset_name = dataset_name[-2]
    print("数据集名称:", dataset_name)
    # 加载真实标签
    # 由于默认 input = data/arenas_combined_edges.txt
    # 最终     true_alignments_fname = data/arenas_edges-mapping-permutation.txt
    true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt"  # can be changed if desired
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

    # Load in attributes if desired (assumes they are numpy array)
    # 如果需要，加载属性（假设它们是numpy数组）

    # None是一个特殊常量，表示一个空的对象。
    # Python当中等于False的值并不只有False一个
    # 对于基本类型来说，基本上每个类型都存在一个值会被判定为False
    # 布尔型，False表示False，其他为True
    # 整数和浮点数，0表示False，其他为True
    # 字符串和类字符串类型（包括bytes和unicode），空字符串表示False，其他为True
    # 序列类型（包括tuple，list，dict，set等），空表示False，非空表示True
    # None永远表示False

    # 如果指定了属性目录
    if args.attributes is not None:
        args.attributes = np.load(args.attributes)  # 从文件加载属性向量
        print("属性向量读取成功！")
        print(args.attributes.shape)

    # Learn embeddings and save to output
    # 学习嵌入与保存输出
    print("开始表示学习中")
    before_rep = time.time()
    embed = learn_representations(args)
    print(embed)
    after_rep = time.time()
    print("表示学习用时 %f 秒" % (after_rep - before_rep))

    emb1, emb2 = get_embeddings(embed)
    before_align = time.time()
    if args.numtop == 0:
        args.numtop = None
    alignment_matrix = get_embedding_similarities(emb1, emb2, num_top=None)  # args.numtop)

    # Report scoring and timing
    # 报告得分和用时
    after_align = time.time()
    total_time = after_align - before_align
    print("对齐用时: "), total_time

    if true_alignments is not None:
        topk_scores = [1]  # ,5,10,20,50]
        for k in topk_scores:
            score, correct_nodes = score_alignment_matrix(alignment_matrix, topk=k, true_alignments=true_alignments)
            print("k=%d: score=%f" % (k, score))


# Should take in a file with the input graph as edgelist (args.input)
# Should save representations to args.output
# 应将输入图形写入一个边缘列表文件 args.input，然后读取
# 将表示保存到 args.output
# 表示学习

def learn_representations(args):
    nx_graph = nx.read_edgelist(args.input, nodetype=int, comments="%")
    print("读入图")
    adj = nx.adjacency_matrix(nx_graph, nodelist=range(nx_graph.number_of_nodes()))
    print("图的邻接矩阵表示")
    # 创建Graph对象
    graph = Graph(adj, node_attributes=args.attributes)
    max_layer = args.untillayer  # xNetMF层之前的计算，默认为2
    if args.untillayer == 0:
        max_layer = None
    alpha = args.alpha          # 折现因子，默认为0.01
    num_buckets = args.buckets  # degree binning的对数基  BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    # 创建RepMethod对象
    rep_method = RepMethod(max_layer=max_layer,          # xNetMF层之前的计算,			 	  默认为2
                           alpha=alpha,                  # 折现因子,						      默认为0.01
                           k=args.k,                     # 控制要采样的地标(landmark)数量，	  默认值为10
                           num_buckets=num_buckets,      # 度（节点特征）装箱(degree binning)的对数基
                           normalize=True,               # 是否标准化
                           gammastruc=args.gammastruc,   # 结构相似性权重，					  默认为1
                           gammaattr=args.gammaattr)     # 属性相似性权重， 					  默认为1
    if max_layer is None:
        max_layer = 1000
    # print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
    print("最大层=%d 折现因子=%f,进行表示学习" % (max_layer, alpha))
    # 表示学习
    representations = xnetmf.get_representations(graph, rep_method)
    np.save(args.output, representations)
    return representations


if __name__ == "__main__":
    args = parse_args()
    main(args)
