import numpy as np

# 定义表示方法类
class RepMethod():
	def __init__(self, 
				align_info = None, 
				p=None, 
				k=10, 
				max_layer=None, 
				alpha = 0.1, 
				num_buckets = None, 
				normalize = True, 
				gammastruc = 1, 
				gammaattr = 1):
		self.p = p        # sample p points      采样p个点
		self.k = k        # control sample size  控制样本大小
		self.max_layer = max_layer # furthest hop distance up to which to compare neighbors 比较邻居的最远跳跃距离
		self.alpha = alpha         # discount factor for higher layers                      较高层的折扣系数
		self.num_buckets = num_buckets # number of buckets to split node feature values into # CURRENTLY BASE OF LOG SCALE 当前对数刻度基础
		self.normalize = normalize     # whether to normalize node embeddings
		self.gammastruc = gammastruc   # parameter weighing structural similarity in node identity
		self.gammaattr = gammaattr     # parameter weighing attribute similarity in node identity

# 定义图类
class Graph():
	#Undirected, unweighted
	def __init__(self, 
				adj,                # 邻接矩阵
				num_buckets=None, 
				node_labels = None, 
				edge_labels = None,
				graph_label = None, 
				node_attributes = None, 
				true_alignments = None):
		self.G_adj = adj             # 邻接矩阵
		self.N = self.G_adj.shape[0] # 结点数
		self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
		self.max_degree = max(self.node_degrees)
		self.num_buckets = num_buckets # how many buckets to break node features into

		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.graph_label = graph_label
		self.node_attributes = node_attributes # N x A matrix, where N is # of nodes, and A is # of attributes
		self.kneighbors = None                 # dict of k-hop neighbors for each node
		self.true_alignments = true_alignments # dict of true alignments, if this graph is a combination of multiple graphs