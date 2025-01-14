import numpy as np
import scipy.io as sio
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp
from scipy.spatial.distance import cosine

# Embedding 就是用一个数值向量“表示”一个对象（Object）的方法
# 计算embedding之间的相似性
def get_embedding_similarities(embed, embed2 = None, sim_measure = "euclidean", num_top = None):
	n_nodes, dim = embed.shape
	if embed2 is None:
		embed2 = embed

	if num_top is not None: # 仅计算top相似度的KD树  KD tree with only top similarities computed
		kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, num_top = num_top)
		return kd_sim

	# 计算所有成对结点的距离
	if sim_measure == "cosine":		# 余弦距离
		similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
	else:							# 不知道啥距离
		similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
		similarity_matrix = np.exp(-similarity_matrix)

	return similarity_matrix

# Split embeddings in half
# Right now asssume graphs are same size (as done in paper's experiments)
# NOTE: to handle graphs of different sizes, pass in an arbitrary split index
# Similarly, to embed >2 graphs, change to pass in a list of splits and return list of embeddings

# 将总的embedding根据初始所处的图拆分成2个或多个embedding

def get_embeddings(combined_embed, graph_split_idx = None):
	if graph_split_idx is None:
		graph_split_idx = int(combined_embed.shape[0] / 2)
	dim = combined_embed.shape[1]
	embed1 = combined_embed[:graph_split_idx]
	embed2 = combined_embed[graph_split_idx:]

	return embed1, embed2

# alignments are dictionary of the form node_in_graph 1 : node_in_graph2
# rows of alignment matrix are nodes in graph 1, columns are nodes in graph2
# 对齐是一个如下形式的字典：node_in_graph 1 : node_in_graph2
# 对齐矩阵：行是图1中的结点，列是图2中的结点
# 计算对齐的得分
def score(alignment_matrix, true_alignments = None):
	if true_alignments is None:  # assume it's just identity permutation
		return np.sum(np.diagonal(alignment_matrix))
	else:
		nodes_g1 = [int(node_g1) for node_g1 in true_alignments.keys()]
		nodes_g2 = [int(true_alignments[node_g1]) for node_g1 in true_alignments.keys()]
		return np.sum(alignment_matrix[nodes_g1, nodes_g2])


# 计算embedding矩阵得分
def score_embeddings_matrices(embed1, embed2, topk = None, similarity_threshold = None, normalize = False, true_alignments = None, sim="cosine"):
	similarity_matrix = get_embedding_similarities(embed1, embed2, sim_measure = sim)
	alignment_matrix = get_estimated_alignment_matrix(similarity_matrix, similarity_threshold, normalize)
	score = score_alignment_matrix(alignment_matrix, topk = topk, true_alignments = true_alignments)
	return score

# kd数对齐
def kd_align(emb1, emb2, normalize=False, distance_metric = "euclidean", num_top = 50):
	kd_tree = KDTree(emb2, metric = distance_metric)	
		
	row = np.array([])
	col = np.array([])
	data = np.array([])
	
	dist, ind = kd_tree.query(emb1, k = num_top)
	print("查询对齐(queried alignments)")
	row = np.array([])
	for i in range(emb1.shape[0]):
		row = np.concatenate((row, np.ones(num_top)*i))
	col = ind.flatten()
	data = np.exp(-dist).flatten()
	sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
	return sparse_align_matrix.tocsr()


def score_alignment_matrix(alignment_matrix, topk = None, topk_score_weighted = False, true_alignments = None):
	n_nodes = alignment_matrix.shape[0]
	correct_nodes = []

	if topk is None:
		row_sums = alignment_matrix.sum(axis=1)
		row_sums[row_sums == 0] = 1e-6  # 应该不会有太大影响，因为0除以0等于0 (shouldn't affect much since dividing 0 by anything is 0)
		alignment_matrix = alignment_matrix / row_sums[:, np.newaxis] # 归一化 normalize

		alignment_score = score(alignment_matrix, true_alignments = true_alignments)
	else: 
		alignment_score = 0   
		if not sp.issparse(alignment_matrix):
			sorted_indices = np.argsort(alignment_matrix)
		
		for node_index in range(n_nodes):
			target_alignment = node_index   # 默认值：假设身份映射，结点应与自身对齐 default: assume identity mapping, and the node should be aligned to itself 
			if true_alignments is not None: # 如果我们有真正的对齐（这是我们所需要的），请为每个节点使用这些对齐 if we have true alignments (which we require), use those for each node
				target_alignment = int(true_alignments[node_index])
			if sp.issparse(alignment_matrix):
				row, possible_alignments, possible_values = sp.find(alignment_matrix[node_index])
				node_sorted_indices = possible_alignments[possible_values.argsort()]
			else:
				node_sorted_indices = sorted_indices[node_index]
			if target_alignment in node_sorted_indices[-topk:]:
				if topk_score_weighted:
					alignment_score += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
				else:
					alignment_score += 1
				correct_nodes.append(node_index)
		alignment_score /= float(n_nodes)

	return alignment_score, set(correct_nodes)