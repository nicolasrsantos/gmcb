import sys
import numpy
import multiprocessing as mp

from models.similarity import Similarity


def modified_starmap_async(function, kwargs):
    return function(**kwargs)


class Coarsening:

    def __init__(self, source_graph, **kwargs):

        prop_defaults = {
            'reduction_factor': [0.5], 'max_levels': [3], 'matching': ['rgmb'],
            'similarity': ['common_neighbors'], 'itr': [10], 'upper_bound': [0.2], 'seed_priority': ['degree'],
            'gmv': [None], 'tolerance': [0.01], 'reverse': None, 'projection': 'common_neighbors',
            'pgrd': [0.50], 'deltap': [0.35], 'deltav': [0.35], 'wmin': [0.0], 'wmax': [1.0], 'threads': 1
        }

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

        self.source_graph = source_graph
        self.hierarchy_graphs = []
        self.hierarchy_levels = []

        # Validation of list values
        for prop_name, prop_value in prop_defaults.items():
            if prop_name != 'threads' and len(getattr(self, prop_name)) == 1:
                setattr(self, prop_name, [getattr(self, prop_name)[0]] * self.source_graph['layers'])

        # Parameters dimension validation
        for prop_name, prop_value in prop_defaults.items():
            if prop_name not in ['threads', 'projection']:
                if self.source_graph['layers'] != len(getattr(self, prop_name)):
                    print('Number of layers and ' + str(prop_name) + ' do not match.')
                    sys.exit(1)

        if self.threads > mp.cpu_count():
            print('Warning: Number of defined threads (' + str(self.threads) + ') '
                  'cannot be greater than the real number of cors (' + str(mp.cpu_count()) + ').\n The number of '
                  'threads was setted as ' + str(mp.cpu_count()))
            self.threads = mp.cpu_count()
            sys.exit(1)

        # Matching method validation
        valid_matching = ['rgmb', 'gmb', 'mlpb', 'hem', 'lem', 'rm', 'mnmf', 'msvm']
        for index, matching in enumerate(self.matching):
            matching = matching.lower()
            if matching not in valid_matching:
                print('Matching ' + matching + ' method is invalid.')
                sys.exit(1)
            self.matching[index] = matching

        # Seed priority validation
        valid_seed_priority = ['strength', 'degree', 'random']
        for index, seed_priority in enumerate(self.seed_priority):
            seed_priority = seed_priority.lower()
            if seed_priority not in valid_seed_priority:
                print('Seed priotiry ' + seed_priority + ' is invalid.')
                sys.exit(1)
            self.seed_priority[index] = seed_priority

        # Reverse validation
        for index, reverse in enumerate(self.reverse):
            if reverse.lower() in ('yes', 'true', 't', 'y', '1'):
                self.reverse[index] = True
            elif reverse.lower() in ('no', 'false', 'f', 'n', '0'):
                self.reverse[index] = False
            else:
                print('Boolean value expected in -rv.')
                sys.exit(1)

        # Similarity measure validation
        valid_similarity = [
            'common_neighbors', 'weighted_common_neighbors',
            'salton', 'preferential_attachment', 'jaccard', 'weighted_jaccard',
            'adamic_adar', 'resource_allocation', 'sorensen', 'hub_promoted',
            'hub_depressed', 'leicht_holme_newman', 'newman_collaboration',
            'unweight', 'embedding_similarity'
        ]
        for index, similarity in enumerate(self.similarity):
            similarity = similarity.lower()
            if similarity not in valid_similarity:
                print('Similarity ' + similarity + ' measure is invalid.')
                sys.exit(1)
            self.similarity[index] = similarity

        self.projection = self.projection.lower()
        if self.projection not in valid_similarity:
            print('Projection similarity ' + self.projection + ' measure is invalid.')
            sys.exit(1)

        for layer in range(self.source_graph['layers']):
            if self.matching[layer] in ['rgmb', 'gmb', 'hem', 'lem', 'rm', 'mnmf', 'msvm']:
                # if self.gmv[layer] is not None:
                #     self.gmv[layer] = None
                #     text = 'Matching method ' + self.matching[layer]
                #     text += ' (setted in layer '
                #     text += str(layer) + ') does not accept -gmv parameter.'
                #     print(text)
                if self.reduction_factor[layer] is not None and self.reduction_factor[layer] > 0.5:
                    self.reduction_factor[layer] = 0.5
                    text = 'Matching method ' + self.matching[layer]
                    text += ' (setted in layer '
                    text += str(layer) + ') does not accept -rf > 0.5.'
                    print(text)

    def run(self):
        graph = self.source_graph.copy()

        while True:

            level = graph['level']
            contract = False
            if level[0] == self.max_levels[0] or level[1] == self.max_levels[1]:
                break

            args = []
            for layer in range(graph['layers']):

                do_matching = True
                if self.gmv[layer] is None and level[layer] >= self.max_levels[layer]:
                    do_matching = False
                elif self.gmv[layer] and graph['vertices'][layer] <= self.gmv[layer]:
                    do_matching = False

                if do_matching:

                    contract = True
                    level[layer] += 1

                    kwargs = dict(reduction_factor=self.reduction_factor[layer])

                    kwargs['gmv'] = self.gmv[layer]
                    if self.matching[layer] in ['mlpb', 'gmb', 'rgmb']:
                        kwargs['vertices'] = graph['vertices_by_type'][layer]
                        kwargs['reverse'] = self.reverse[layer]
                    if self.matching[layer] in ['mlpb', 'rgmb']:
                        kwargs['seed_priority'] = self.seed_priority[layer]
                    if self.matching[layer] in ['mlpb']:
                        kwargs['upper_bound'] = self.upper_bound[layer]
                        kwargs['n'] = self.source_graph['vertices'][layer]
                        kwargs['tolerance'] = self.tolerance[layer]
                        kwargs['itr'] = self.itr[layer]

                    if self.matching[layer] in ['hem', 'lem', 'rm', 'mnmf', 'msvm']:
                        graph['projection'] = getattr(Similarity(graph, graph['adjlist']), self.projection)
                        one_mode_graph = graph.weighted_one_mode_projection(graph['vertices_by_type'][layer], similarity=self.similarity[layer])
                        matching_function = getattr(one_mode_graph, self.matching[layer])
                    else:
                        graph['similarity'] = getattr(Similarity(graph, graph['adjlist']), self.similarity[layer])
                        matching_function = getattr(graph, self.matching[layer])
                    # Create a args for the engine multiprocessing.pool
                    args.append([(matching_function, kwargs)])

            if contract:
                print("contracting graph")
                # Create pools
                pool = mp.Pool(processes=self.threads)
                processes = []
                for arg in args:
                    processes.append(pool.starmap_async(modified_starmap_async, arg))

                # Merge chunked solutions
                matching = numpy.arange(graph.vcount())
                for process in processes:
                    print("running process")
                    result = process.get()[0]
                    print("running process")
                    vertices = numpy.where(result > -1)[0]
                    print("running process")
                    matching[vertices] = result[vertices]

                # Close processes
                pool.close()
                pool.join()

                # Contract current graph using the matching
                coarsened_graph = graph.contract(matching, self.similarity)
                coarsened_graph['level'] = level
                print("graph contracted and level created.")

                if coarsened_graph.vcount() == graph.vcount():
                    break

                self.hierarchy_graphs.append(coarsened_graph)
                self.hierarchy_levels.append(level[:])
                graph = coarsened_graph
                print("graph contracted")
            else:
                break
