import math

from numpy import dot
from numpy.linalg import norm
from numpy import linalg as LA

class Similarity(object):

	graph, adjlist = (None,) * 2

	def __init__(self, graph, adjlist):
		self.graph = graph
		self.adjlist = adjlist

	def unweight(self, i, j):
		""" Calculates pairwise weight edge on a given graph. """

		return 1.0

	def weight(self, i, j):
		""" Calculates pairwise weight edge on a given graph. """

		return self.graph[i, j]

	def get_common_neighbors(self, i, j):
		""" Calculates pairwise common neighbors similarity on a given unweighted graph. """

		return self.adjlist[i].intersection(self.adjlist[j])

	def nmf_cosine(self, i, j):
		""" The similarity between two nodes is given by the cosine of the
		angle between the corresponding columns in H, multiplied
		by their norms. """

		h_i = self.graph.vs[i]['nmf_array']
		h_j = self.graph.vs[j]['nmf_array']
		cosine = dot(h_i, h_j) / (norm(h_i) * norm(h_j))
		return LA.norm(h_i, 1) * LA.norm(h_j, 1) * cosine

	def preferential_attachment(self, i, j):
		""" Calculates pairwise preferential attachment similarities on a given unweighted graph. """

		return float(self.graph.degree(i)) * float(self.graph.degree(j))

	def common_neighbors(self, i, j):
		""" Calculates pairwise common neighbors similarity on a given unweighted graph. """

		return len(self.adjlist[i].intersection(self.adjlist[j]))

	def newman_collaboration(self, i, j):
		""" Calculates pairwise Newman’s collaboration similarity """

		_sum = 0.0
		cn = self.adjlist[i].intersection(self.adjlist[j])
		for z in cn:
			_sum += 1 / (self.graph.strength(z) - 1)
		return _sum

	def weighted_common_neighbors(self, i, j):
		"""
		Calculates pairwise common neighbors similarity uses the edge-weight information
		on a given unweighted graph.
		"""

		_sum = 0.0
		cn = self.adjlist[i].intersection(self.adjlist[j])
		for z in cn:
			_sum += (self.graph[i, z] + self.graph[j, z]) / 2
		return _sum

	def jaccard(self, i, j):
		""" Calculates pairwise jaccard similarity on a given unweighted graph. """

		isect = len(self.adjlist[i].intersection(self.adjlist[j]))
		union = (len(self.adjlist[i]) + len(self.adjlist[j]) - isect)
		return 0 if union == 0 else isect / float(union)

	def weighted_jaccard(self, i, j):
		""" Calculates pairwise jaccard similarity on a given unweighted graph. """

		_sum_isect = 0.0
		isect = self.adjlist[i].intersection(self.adjlist[j])
		for z in isect:
			_sum_isect += (self.graph[i, z] + self.graph[j, z]) / 2

		_sum_union = 0.0
		union = self.adjlist[i].union(self.adjlist[j]) - isect
		for z in union:
			_sum_union += (self.graph[i, z] + self.graph[j, z]) / 2

		return 0 if _sum_union == 0.0 else _sum_isect / _sum_union

	def salton(self, i, j):
		""" Calculates pairwise solton similarity on a given unweighted graph. """

		product = float(self.graph.degree(i)) * float(self.graph.degree(j))
		if product == 0.0:
			return 0.0

		isect = len(self.adjlist[i].intersection(self.adjlist[j]))
		return isect / math.sqrt(product)

	def adamic_adar(self, i, j):
		""" Calculates pairwise adamic adar similarity on a given unweighted graph. """

		score = 0.0
		for isect in self.adjlist[i].intersection(self.adjlist[j]):
			degree = self.graph.degree(isect)
			if degree != 0:
				score += 1 / math.log(degree)
		return score

	def resource_allocation(self, i, j):
		""" Calculates pairwise resource allocation similarity on a given unweighted graph. """

		score = 0.0
		for isect in self.adjlist[i].intersection(self.adjlist[j]):
			degree = self.graph.degree(isect)
			if degree != 0:
				score += 1 / degree
		return score

	def sorensen(self, i, j):
		""" Calculates pairwise sorensen similarity on a given unweighted graph. """

		_sum = float(self.graph.degree(i)) * float(self.graph.degree(j))
		if _sum == 0.0:
			return 0.0

		isect = 2 * len(self.adjlist[i].intersection(self.adjlist[j]))
		return isect / _sum

	def hub_promoted(self, i, j):
		""" Calculates pairwise hub promoted similarity on a given unweighted graph. """

		minimum = min(float(self.graph.degree(i)), float(self.graph.degree(j)))
		if minimum == 0.0:
			return 0.0

		isect = len(self.adjlist[i].intersection(self.adjlist[j]))
		return isect / minimum

	def hub_depressed(self, i, j):
		""" Calculates pairwise hub depressed similarity on a given unweighted graph. """

		maximum = max(float(self.graph.degree(i)), float(self.graph.degree(j)))
		if maximum == 0.0:
			return 0.0

		isect = len(self.adjlist[i].intersection(self.adjlist[j]))
		return isect / maximum

	def leicht_holme_newman(self, i, j):
		""" Calculates pairwise leicht holmeNewman similarity on a given unweighted graph. """

		product = float(self.graph.degree(i)) * float(self.graph.degree(j))
		if product == 0.0:
			return 0.0

		isect = len(self.adjlist[i].intersection(self.adjlist[j]))
		return isect / product

	def within_common_neighbors(self, i, j):
		"""
		Calculates pairwise inter-cluster common neighbors (WCN) similarity on a
		given unweighted graph. WCN considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		return within_isect

	def within_jaccard(self, i, j):
		"""
		Calculates pairwise inter-cluster jaccard (WJac) similarity on a
		given unweighted graph. WJac considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		union = (len(self.adjlist[i]) + len(self.adjlist[j]) - within_isect)
		return 0 if union == 0 else within_isect / float(union)

	def within_salton(self, i, j):
		"""
		Calculates pairwise inter-cluster salton (WSal) similarity on a
		given unweighted graph. WSal considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		product = float(self.graph.degree(i)) * float(self.graph.degree(j))

		if product == 0.0:
			return 0.0

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		return within_isect / math.sqrt(product)

	def within_adamic_adar(self, i, j):
		"""
		Calculates pairwise inter-cluster adamic adar (WAA) similarity on a
		given unweighted graph. WAA considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		score = 0.0
		for isect in self.adjlist[i].intersection(self.adjlist[j]):
			if self.vs[isect]['membership'] == self.vs[i]['membership']:
				degree = self.graph.degree(isect)
				if degree != 0:
					score += 1 / math.log(degree)
		return score

	def within_resource_allocation(self, i, j):
		"""
		Calculates pairwise inter-cluster resource allocation (WRL) similarity on a
		given unweighted graph. WRL considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		score = 0.0
		for isect in self.adjlist[i].intersection(self.adjlist[j]):
			if self.vs[isect]['membership'] == self.vs[i]['membership']:
				degree = self.graph.degree(isect)
				if degree != 0:
					score += 1 / degree
		return score

	def within_sorensen(self, i, j):
		"""
		Calculates pairwise inter-cluster sorensen (WSor) similarity on a
		given unweighted graph. WSor considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		_sum = float(self.graph.degree(i)) * float(self.graph.degree(j))
		if _sum == 0.0:
			return 0.0

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 2.0
		return within_isect / _sum

	def within_hub_promoted(self, i, j):
		"""
		Calculates pairwise inter-cluster hub promoted (WHP) similarity on a
		given unweighted graph. WHP considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		minimum = min(float(self.graph.degree(i)), float(self.graph.degree(j)))
		if minimum == 0.0:
			return 0.0

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		return within_isect / minimum

	def within_hub_depressed(self, i, j):
		"""
		Calculates pairwise inter-cluster hub depressed (WHD) similarity on a
		given unweighted graph. WHD considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		maximum = max(float(self.graph.degree(i)), float(self.graph.degree(j)))
		if maximum == 0.0:
			return 0.0

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		return within_isect / maximum

	def within_leicht_holme_newman(self, i, j):
		"""
		Calculates pairwise inter-cluster leicht holme newman (WLHN) similarity on a
		given unweighted graph. WLHN considering solely the subset of within-cluster
		common neighbors instead of the set of all common neighbors
		"""

		product = float(self.graph.degree(i)) * float(self.graph.degree(j))
		if product == 0.0:
			return 0.0

		isect = self.adjlist[i].intersection(self.adjlist[j])
		within_isect = 0.0
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				within_isect += 1.0
		return within_isect / product

	def wic(self, i, j):
		"""
		Calculates similarity between a pair of vertices using information from
		intra-cluster or within-cluster (W) and inter-cluster (IC) common
		neighbors of these vertices.
		"""

		isect = self.adjlist[i].intersection(self.adjlist[j])
		nWcn = 0.0 # Intra cluster or intra community
		nIcn = 0.0 # Inter clusters or inter comunities
		for vertex in isect:
			if self.vs[vertex]['membership'] == self.vs[i]['membership']:
				nWcn += 1.0
			else:
				nIcn += 1.0
		return nWcn if (nIcn == 0.0) else nWcn / nIcn

	def embedding_similarity(self, i, j):
		"""
		Computes the cosine similarity between the embedding vectors of node i and j.
		"""

		emb_i = self.graph['sv_embeddings'][i]
		emb_j = self.graph['sv_embeddings'][j]
		vec_dot = dot(emb_i, emb_j)
		vec_norm = norm(emb_i) * norm(emb_j)
		sim = vec_dot / vec_norm
		return sim
