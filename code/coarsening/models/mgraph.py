import operator

import numpy
import random
import math
import collections
import pickle as pkl
import time

from random import sample
from igraph import Graph
from scipy import sparse
from numpy import dot, mean
from numpy.linalg import norm
from numpy import linalg as LA
from models.similarity import Similarity
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.decomposition import non_negative_factorization
# from sklearn.decomposition import ProjectedGradientNMF
from sklearn.decomposition import NMF
import warnings

def load_ncol(filename):
    """
    Load ncol npartite graph and generate special attributes
    """

    data = numpy.loadtxt(filename, skiprows=0, dtype=str)
    dict_edges = dict()
    for row in data:
        if len(row) == 3:
            dict_edges[(int(row[0]), int(row[1]))] = float(row[2])
        else:
            dict_edges[(int(row[0]), int(row[1]))] = 1

    edges, weights = list(zip(*dict_edges.items()))
    return edges, weights


def load_embeddings(doc_emb_filename, word_emb_filename):
    """
    Load document and word embedding files
    """

    with open(doc_emb_filename, 'rb') as f:
        embeddings = pkl.load(f)
    with open(word_emb_filename, 'rb') as f:
        word_embeddings = pkl.load(f)
    embeddings.extend(word_embeddings)
    return embeddings


class MGraph(Graph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(
            self,
            network_filename,
            vertices,
            similarity,
            word_emb_filename=None,
            doc_emb_filename=None,
            y_filename=None,
            data_split=None,
            filename_type='ncol'
        ):
        """
        filename_type: ncol, arff
        """
        print("loading data...")
        edges, weights = None, None
        if filename_type == 'ncol':
            edges, weights = load_ncol(network_filename)

        self.add_vertices(sum(vertices))
        self.add_edges(edges)
        self['adjlist'] = list(map(set, self.get_adjlist()))
        self['vertices'] = vertices
        self['layers'] = len(vertices)
        self['level'] = [0] * self['layers']
        self['similarity'] = None
        self['whole_graph_layer1_size'] = vertices[0]
        self['whole_graph_layer2_size'] = vertices[1]
        self.es['weight'] = weights
        types = []
        for layer in range(self['layers']):
            types += [layer] * vertices[layer]
        self.vs['type'] = types
        for v in self.vs():
            v['weight'] = 1
            v['source'] = [v.index]
            v['name'] = [v.index]
            v['predecessor'] = [v.index]
            v['successor'] = [None]

        self['vertices_by_type'] = []
        for layer in range(self['layers']):
            self['vertices_by_type'].append(self.vs.select(type=layer).indices)

        # Not allow direct graphs
        if self.is_directed():
            self.to_undirected(combine_edges=None)

        if 'embedding_similarity' in similarity:
            if word_emb_filename is None or doc_emb_filename is None:
                raise ValueError('embedding filename was not set')

            embeddings = load_embeddings(doc_emb_filename, word_emb_filename)
            self['embeddings'] = embeddings
            self['sv_embeddings'] = embeddings

            if y_filename:
                with open(y_filename, 'rb') as f:
                    self['y'] = pkl.load(f)
                    self['sv_y'] = self['y']

            if data_split:
                with open(data_split, 'rb') as f:
                    self['splits'] = pkl.load(f)
                    self['sv_splits'] = self['splits']
        print("data loaded.")

    def contract(self, matching, similarity):
        """
        Create coarse graph from matching of groups
        """
        print("contracting graph")
        # Contract vertices: Referencing the original graph of the coarse graph
        types = []
        weights = []
        sources = []
        predecessors = []
        matching = numpy.array(matching)
        uniqid = 0
        for layer in range(self['layers']):
            print(f"contracting layer {layer}")
            start = sum(self['vertices'][0:layer]) # inicio da camada|  ex:    0
            end = sum(self['vertices'][0:layer + 1]) # fim da camada|  até  7674
            matching_layer = matching[start:end]
            vertices_layer = numpy.arange(start, end)
            clusters = numpy.unique(matching_layer)
            for cluster_id in clusters:
                print(f"working on cluster {cluster_id}")
                ids = numpy.where(matching_layer == cluster_id)[0]
                vertices = vertices_layer[ids]
                weight = 0
                source = []
                predecessor = []
                for vertex in vertices:
                    print(f"inside cluster - working on vertex {vertex}")
                    self.vs[vertex]['successor'] = uniqid
                    weight += self.vs[vertex]['weight']
                    source.extend(self.vs[vertex]['source'])
                    predecessor.append(vertex)
                weights.append(weight)
                types.append(layer)
                sources.append(source)
                predecessors.append(predecessor)
                uniqid += 1

        print("DONE CONTRACTING ________________________________")
        # Create coarsened version
        coarse = MGraph()
        coarse.add_vertices(uniqid)
        coarse.vs['type'] = types
        coarse.vs['weight'] = weights
        coarse.vs['name'] = range(coarse.vcount())
        coarse.vs['successor'] = [None] * coarse.vcount()
        coarse.vs['source'] = sources
        coarse.vs['predecessor'] = predecessors
        coarse['layers'] = self['layers']
        coarse['similarity'] = None
        coarse['vertices'] = []

        coarse['vertices_by_type'] = []
        for layer in range(self['layers']):
            coarse['vertices_by_type'].append(coarse.vs.select(type=layer).indices)
            coarse['vertices'].append(len(coarse['vertices_by_type'][layer]))

        # Contract edges
        dict_edges = dict()
        for edge in self.es():
            v_successor = self.vs[edge.tuple[0]]['successor']
            u_successor = self.vs[edge.tuple[1]]['successor']

            # Add edge in coarsened graph
            if v_successor < u_successor:
                dict_edges[(v_successor, u_successor)] = dict_edges.get((v_successor, u_successor), 0) + edge['weight']
            else:
                dict_edges[(u_successor, v_successor)] = dict_edges.get((u_successor, v_successor), 0) + edge['weight']

        if len(dict_edges) > 0:
            edges, weights = list(zip(*dict_edges.items()))
            coarse.add_edges(edges)
            coarse.es['weight'] = weights
            coarse['adjlist'] = list(map(set, coarse.get_adjlist()))

        if 'embedding_similarity' in similarity:
            print("embedding similarity")
            sv_embeddings, sv_y, sv_splits = [], [], []
            for source in coarse.vs['source']:
                print(f"WORKING ON SOURCE {source}/{len(coarse.vs['source'])}")
                embs = []
                for v in source:
                    embs.append(self['embeddings'][v])
                embs = mean(embs, axis=0)
                sv_embeddings.append(embs)
                if source[0] < self['whole_graph_layer1_size']:
                    sv_y.append(self['y'][source[0]])
                    sv_splits.append(self['splits'][source[0]])
                print("BUILT EMBEDDINGS")
            coarse['sv_embeddings'] = sv_embeddings
            coarse['embeddings'] = self['embeddings']
            coarse['y'] = self['y']
            coarse['sv_y'] = sv_y
            coarse['splits'] = self['splits']
            coarse['sv_splits'] = sv_splits
            coarse['whole_graph_layer1_size'] = self['whole_graph_layer1_size']
            coarse['whole_graph_layer2_size'] = self['whole_graph_layer2_size']

        return coarse

    def gmb(self, vertices=None, reduction_factor=0.5, reverse=True, gmv=None):
        """
        Matches are restricted between vertices that are not adjacent
        but are only allowed to match with neighbors of its neighbors,
        i.e. two-hops neighborhood
        """
        print("running GMB.")
        matching = numpy.array([-1] * self.vcount())
        matching[vertices] = vertices
        # Search two-hops neighborhood for each vertex in selected layer
        dict_edges = dict()
        visited = [0] * self.vcount()
        doc_layer_size = sum(self['vertices'][:1])
        for vertex in vertices:
            print(f"running GMB for vertex {vertex}")
            neighborhood = self.neighborhood(vertices=vertex, order=2)
            twohops = neighborhood[(len(self['adjlist'][vertex]) + 1):]
            for twohop in twohops:
                if visited[twohop] == 1:
                    continue
                if vertex < doc_layer_size and twohop < doc_layer_size:
                    if (
                        self['sv_y'][vertex] != self['sv_y'][twohop]
                        or self['sv_splits'][vertex] != self['sv_splits'][twohop]
                    ):
                        continue
                dict_edges[(vertex, twohop)] = self['similarity'](vertex, twohop)
            visited[vertex] = 1
        # Select promising matches or pair of vertices
        visited = [0] * self.vcount()
        edges = sorted(dict_edges.items(), key=operator.itemgetter(1), reverse=reverse)
        merge_count = int(reduction_factor * len(vertices))
        if gmv is not None:
            while True:
                print("while true")
                if len(vertices) - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * len(vertices))
        for edge, value in edges:
            print(f"working on edges - {edge}, {value}")
            if merge_count == 0:
                break
            vertex = edge[0]
            neighbor = edge[1]
            if (visited[vertex] != 1) and (visited[neighbor] != 1):
                matching[neighbor] = vertex
                matching[vertex] = vertex
                visited[neighbor] = 1
                visited[vertex] = 1
                merge_count -= 1

        print("DONE GMB -----------------------------------------------")
        return matching

    def rgmb(self, vertices=None, reduction_factor=0.5, seed_priority='random', reverse=True, gmv=None):
        """
        Matches are restricted between vertices that are not adjacent
        but are only allowed to match with neighbors of its neighbors,
        i.e. two-hopes neighborhood. This version use a random seed.
        """

        matching = numpy.array([-1] * self.vcount())
        matching[vertices] = vertices

        # Select seed set expansion
        if seed_priority == 'strength':
            vertices_score = numpy.array(self.strength(vertices, weights='weight'))
            dictionary = dict(zip(vertices, vertices_score))
            vertices_id = sorted(dictionary, key=dictionary.__getitem__, reverse=reverse)
        if seed_priority == 'degree':
            vertices_score = numpy.array(self.degree(vertices))
            dictionary = dict(zip(vertices, vertices_score))
            vertices_id = sorted(dictionary, key=dictionary.__getitem__, reverse=reverse)
        if seed_priority == 'random':
            vertices_id = vertices
            vertices_id = random.sample(vertices_id, len(vertices_id))

        # Find the matching
        visited = [0] * self.vcount()
        index = 0
        merge_count = int(reduction_factor * len(vertices))
        if gmv is not None:
            while True:
                if len(vertices) - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * len(vertices))

        while merge_count > 0 and index < len(vertices):
            # Randomly select a vertex v of V
            vertex = vertices_id[index]
            if visited[vertex] == 1:
                index += 1
                continue
            # Select the edge (v, u) of E which maximum score
            # Tow hopes restriction: It ensures that the match only occurs
            # between vertices of the same type
            neighborhood = self.neighborhood(vertices=vertex, order=2)
            twohops = neighborhood[(len(self['adjlist'][vertex]) + 1):]
            # twohops = set((twohop for onehop in self['adjlist'][vertex] for twohop in self['adjlist'][onehop])) -
            # set([vertex])
            _max = 0.0
            neighbor = vertex
            for twohop in twohops:
                if visited[twohop] == 1:
                    continue
                # Calling a function of a module from a string
                score = self['similarity'](vertex, twohop)
                if score > _max:
                    _max = score
                    neighbor = twohop
            matching[neighbor] = vertex
            matching[vertex] = vertex
            visited[neighbor] = 1
            visited[vertex] = 1
            merge_count -= 1
            index += 1

        return matching

    def rm(self, reduction_factor=0.5, gmv=None):
        """
        Random Matching: Select a maximal matching using a
        randomized algorithm
        """

        matching = numpy.array([-1] * self['source_vertices'])
        merge_count = int(reduction_factor * self.vcount())
        if gmv is not None:
            while True:
                if self.vcount() - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * self.vcount())
        self.get_random_edges(merge_count, matching)
        return matching

    def lem(self, reduction_factor=0.5, gmv=None):
        """
        Heavy Light Matching: Search for a minimal matching using the
        weights of the edges of the graph.
        """

        matching = numpy.array([-1] * self['source_vertices'])
        merge_count = int(reduction_factor * self.vcount())
        if gmv is not None:
            while True:
                if self.vcount() - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * self.vcount())
        self.get_sorted_edges(merge_count, matching, reverse=False)
        return matching

    def mnmf(self, reduction_factor=0.5, k=100, gmv=None):
        """
        Matching via non-negative matrix factorization
        """

        N = self.vcount()
        edges = self.get_edgelist()
        weights = self.es['weight']
        X = sparse.csr_matrix((weights, zip(*edges)), shape=(N, N))

        model = NMF(n_components=k, init='random', random_state=0, max_iter=200, tol=0.005, solver='mu')
        W = model.fit_transform(X)
        H = model.components_

        weights = []
        for edge in self.es():
            a = W[edge.tuple[0]]
            b = W[edge.tuple[1]]
            if not numpy.count_nonzero(a) or not numpy.count_nonzero(b):
                weights.append(0.0)
                continue
            cosine = 1 - spatial.distance.cosine(a, b)
            weights.append(cosine)

        self.es['weight'] = weights
        return self.hem(reduction_factor=reduction_factor, gmv=gmv)

    def msvm(self, reduction_factor=0.5, gmv=None):
        """
        Most Similar Vertex Matching: The algorithm matches the most similar pair based
        on a similar similarity measure, like CN
        """

        matching = numpy.array([-1] * self['source_vertices'])
        vertices = range(self.vcount())
        vertices_id = random.sample(vertices, len(vertices))

        # Find the matching
        visited = [0] * self.vcount()
        index = 0
        merge_count = int(reduction_factor * len(vertices))
        if gmv is not None:
            while True:
                if len(vertices) - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * len(vertices))

        while merge_count > 0 and index < len(vertices):
            # Randomly select a vertex v of V
            vertex = vertices_id[index]
            if visited[vertex] == 1:
                index += 1
                continue
            # Select the edge (v, u) of E which maximum score
            neighbors = self.neighbors(vertex)
            _max = 0.0
            neighbor = vertex
            for n in neighbors:
                if visited[n] == 1:
                    continue
                # Calling a function of a module from a string
                score = self['similarity'](vertex, n)
                if score > _max:
                    _max = score
                    neighbor = n
            # Match vertex and its neighbor with maximum score
            matching[self.vs[neighbor]['name']] = self.vs[vertex]['name']
            matching[self.vs[vertex]['name']] = self.vs[vertex]['name']
            visited[neighbor] = 1
            visited[vertex] = 1
            merge_count -= 1
            index += 1

        return matching

    def hem(self, reduction_factor=0.5, gmv=None):
        """
        Heavy Edge Matching: Search for a maximal matching using the
        weights of the edges of the graph.
        """

        matching = numpy.array([-1] * self['source_vertices'])
        merge_count = int(reduction_factor * self.vcount())
        if gmv is not None:
            while True:
                if self.vcount() - merge_count >= gmv or reduction_factor <= 0.0:
                    break
                reduction_factor -= 0.01
                merge_count = int(reduction_factor * self.vcount())
        self.get_sorted_edges(merge_count, matching, reverse=True)
        return matching

    def get_random_edges(self, merge_count, matching):
        """
        Return a random independent edge set in a graph, i.e., is a set
        of edges without common vertices random selected.
        """

        visited = [0] * self.vcount()
        edges = sample(list(self.es()), self.ecount())
        for edge in edges:
            if merge_count == 0:
                break
            if (visited[edge.tuple[0]] == 0) and (visited[edge.tuple[1]] == 0):
                u = self.vs[edge.tuple[0]]['name']
                v = self.vs[edge.tuple[1]]['name']
                matching[u] = u
                matching[v] = u
                visited[edge.tuple[1]] = 1
                visited[edge.tuple[0]] = 1
                merge_count -= 1

    def get_sorted_edges(self, merge_count, matching, reverse=True):
        """
        Search for a maximal matching using the weights of the edges of
        the graph. The aim is to find a maximal matching of the graph that
        minimizes the cut.
        """

        visited = [0] * self.vcount()
        edges = sorted(self.es(), key=lambda edge: edge['weight'], reverse=reverse)
        for edge in edges:
            if merge_count == 0:
                break
            if (visited[edge.tuple[0]] == 0) and (visited[edge.tuple[1]] == 0):
                u = self.vs[edge.tuple[0]]['name']
                v = self.vs[edge.tuple[1]]['name']
                matching[u] = u
                matching[v] = u
                visited[edge.tuple[1]] = 1
                visited[edge.tuple[0]] = 1
                merge_count -= 1

    def weighted_one_mode_projection(self, vertices, similarity='common_neighbors'):
        """
        Application of a one-mode projection to a bipartite network generates
        two unipartite networks, one for each layer, so that vertices with
        common neighbors are connected by edges in their respective projection.
        """

        graph = MGraph()
        graph.add_vertices(vertices)
        graph['source_vertices'] = self.vcount()
        graph['source_edges'] = self.ecount()
        graph.vs['name'] = self.vs[vertices]['name']
        name_to_id = dict(zip(vertices, range(graph.vcount())))

        dict_edges = dict()
        visited = [0] * self.vcount()
        for vertex in vertices:
            neighborhood = self.neighborhood(vertices=vertex, order=2)
            twohops = neighborhood[(len(self['adjlist'][vertex]) + 1):]
            for twohop in twohops:
                if visited[twohop] == 1:
                    continue
                dict_edges[(name_to_id[vertex], name_to_id[twohop])] = self['projection'](vertex, twohop)
            visited[vertex] = 1

        if len(dict_edges) > 0:
            edges, weights = list(zip(*dict_edges.items()))
            graph.add_edges(edges)
            graph.es['weight'] = weights

        graph['adjlist'] = list(map(set, graph.get_adjlist()))
        graph['similarity'] = getattr(Similarity(graph, graph['adjlist']), similarity)

        return graph

    def mlpb(self, vertices=None, seed_priority='strength', reduction_factor=0.5, itr=10, tolerance=0.05,
             upper_bound=0.2, n=None, gmv=None, reverse=True):

        """ Matching via weight-constrained label propagation and neighborhood. """
        print("running MLPB")
        matching = numpy.array([-1] * self.vcount())
        matching[vertices] = vertices

        if gmv:
            min_vertices = gmv
        else:
            min_vertices = int((1 - reduction_factor) * len(vertices))
        if min_vertices < 1:
            min_vertices = 1

        max_size = int(math.ceil(((1.0 + upper_bound) * n) / min_vertices))
        number_of_vertices = len(vertices)
        weight_of_sv = self.vs['weight']
        label_dict = dict(zip(vertices, vertices))
        twohops_dict = collections.defaultdict(int)
        similarity_dict = collections.defaultdict(float)

        # Select seed set expansion: case of strength or degree seed
        if seed_priority == 'strength':
            vertices_score = numpy.array(self.strength(vertices, weights='weight'))
            dictionary = dict(zip(vertices, vertices_score))
            vertices_id = sorted(dictionary, key=dictionary.__getitem__, reverse=reverse)
        if seed_priority == 'degree':
            vertices_score = numpy.array(self.degree(vertices))
            dictionary = dict(zip(vertices, vertices_score))
            vertices_id = sorted(dictionary, key=dictionary.__getitem__, reverse=reverse)

        tolerance = tolerance * len(vertices)
        swap = tolerance + 1

        while (tolerance < swap) and itr:
            swap = 0
            itr -= 1

            # Select seed set expansion: case of random seed
            if seed_priority == 'random':
                vertices_id = vertices
                vertices_id = random.sample(vertices_id, len(vertices_id))

            for vertex in vertices_id:

                if self.degree(vertex) == 0:
                    continue

                # Tow hopes restriction: It ensures that the match only occurs
                # between vertices of the same type
                if not twohops_dict.get(vertex, False):
                    neighborhood = self.neighborhood(vertices=vertex, order=2)
                    twohops_dict[vertex] = neighborhood[(len(self['adjlist'][vertex]) + 1):]

                # Update neighborhood edge density
                Q = collections.defaultdict(float)
                for neighbor in twohops_dict[vertex]:
                    if weight_of_sv[label_dict[neighbor]] + self.vs[vertex]['weight'] <= max_size:
                        if vertex < neighbor:
                            u, v = vertex, neighbor
                        else:
                            u, v = neighbor, vertex
                        if not similarity_dict.get((u, v), False):
                            # similarity_dict[(u, v)] = self['similarity'](u, v) / math.sqrt(self.degree(u) + self.degree(v))
                            similarity_dict[(u, v)] = self['similarity'](u, v)
                        if similarity_dict[(u, v)] > 0.0:
                            Q[label_dict[neighbor]] += similarity_dict[(u, v)]

                total_similarity = sum(Q.values())
                for li in Q.keys():
                    Q[li] -= (total_similarity - Q[li])

                if Q:
                    # Select the dominant label
                    dominant_label = max(Q.items(), key=operator.itemgetter(1))[0]
                    prev_label = label_dict[vertex]
                    # If a dominant label was fund, match them together
                    # and update data structures
                    if dominant_label != prev_label:
                        swap += 1
                        # Update vertex label
                        label_dict[vertex] = dominant_label
                        # Update the super-vertex weight
                        weight_of_sv[prev_label] -= self.vs[vertex]['weight']
                        weight_of_sv[dominant_label] += self.vs[vertex]['weight']
                        # Verify the size-constraint restriction
                        if weight_of_sv[prev_label] == 0:
                            number_of_vertices -= 1
                        if number_of_vertices <= min_vertices:
                            tolerance = swap
                            break

        for key, value in label_dict.items():
            matching[key] = value

        return matching

    def number_of_components(self):
        components = self.components()
        components_sizes = components.sizes()
        return len(components_sizes)
