## 导入第三方库
import numpy as np
import argparse									# 导入argparse包
# argparse 模块是 Python 内置的用于命令项选项与参数解析的模块，
# argparse 模块可以让人轻松编写用户友好的命令行接口，能够帮助程序员为模型定义参数。
import networkx as nx
import time
import os
import sys
try: import cPickle as pickle 
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
	parser = argparse.ArgumentParser(description="Run REGAL.") 				# 2 创建一个命令行解析器对象
																			# 3 给解析器添加命令行参数
	parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt', help="Edgelist of combined input graph")

	parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
	                    help='Embeddings path')

	parser.add_argument('--attributes', nargs='?', default=None,
	                    help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

	parser.add_argument('--attrvals', type=int, default=2,
	                    help='Number of attribute values. Only used if synthetic attributes are generated')


	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--k', type=int, default=10,
	                    help='Controls of landmarks to sample. Default is 10.')

	parser.add_argument('--untillayer', type=int, default=2,
                    	help='Calculation until the layer for xNetMF.')
	parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
	parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
	parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
	parser.add_argument('--numtop', type=int, default=10,help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
	parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
	return parser.parse_args()           									# 4 解析命令行的参数 ——使用 parse_args() 解析添加的参数

def main(args):
	dataset_name = args.output.split("/")
	if len(dataset_name) == 1:
		dataset_name = dataset_name[-1].split(".")[0]
	else:
		dataset_name = dataset_name[-2]

	# 加载真实标签
	true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" #can be changed if desired
	print("真实标签文件目录: ", true_alignments_fname)
	true_alignments = None
	if os.path.exists(true_alignments_fname):
		print("真实标签文件存在！")
		with open(true_alignments_fname, "rb") as true_alignments_file:
			try:
				true_alignments = pickle.load(true_alignments_file)
			except:
				true_alignments = pickle.load(true_alignments_file, encoding = "latin1")
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

	if args.attributes is not None:
		args.attributes = np.load(args.attributes) # 从文件加载属性向量
		print("属性向量读取成功！")
		print(args.attributes.shape)

	# Learn embeddings and save to output
	# 学习嵌入与保存输出
	print("学习表征中...")
	before_rep = time.time()
	embed = learn_representations(args)
	print(embed)
	after_rep = time.time()
	print("学习表征用时 %f 秒" % (after_rep - before_rep))

	emb1, emb2 = get_embeddings(embed)
	before_align = time.time()
	if args.numtop == 0:
		args.numtop = None
	alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = None)#args.numtop)

	# Report scoring and timing
	# 报告得分和用时
	after_align = time.time()
	total_time = after_align - before_align
	print("对齐用时: "), total_time

	if true_alignments is not None:
		topk_scores = [1]#,5,10,20,50]
		for k in topk_scores:
			score, correct_nodes = score_alignment_matrix(alignment_matrix, topk = k, true_alignments = true_alignments)
			print("得分 %d: %f" % (k, score))

# Should take in a file with the input graph as edgelist (args.input)
# Should save representations to args.output
# 应将输入图形写入一个边缘列表文件 args.input，然后读取
# 将表示保存到 args.output
# 学习特征

def learn_representations(args):
	nx_graph = nx.read_edgelist(args.input, nodetype = int, comments="%")
	print("读入图")
	adj = nx.adjacency_matrix(nx_graph, nodelist = range(nx_graph.number_of_nodes()) )
	print("得到邻接矩阵")
	
	graph = Graph(adj, node_attributes = args.attributes)
	max_layer = args.untillayer
	if args.untillayer == 0:
		max_layer = None
	alpha = args.alpha
	num_buckets = args.buckets #BASE OF LOG FOR LOG SCALE
	if num_buckets == 1:
		num_buckets = None
	rep_method = RepMethod(max_layer = max_layer, 
							alpha = alpha, 
							k = args.k, 
							num_buckets = num_buckets, 
							normalize = True, 
							gammastruc = args.gammastruc, 
							gammaattr = args.gammaattr)
	if max_layer is None:
		max_layer = 1000
	print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
	representations = xnetmf.get_representations(graph, rep_method)
	np.save(args.output, representations)
	return representations		

if __name__ == "__main__":
	args = parse_args()
	main(args)
