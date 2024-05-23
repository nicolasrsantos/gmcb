import sys
import numpy
import os
import inspect
import json

from models.mgraph import MGraph
from models.coarsening import Coarsening
import models.args as args

from models.timing import Timing

def main():
    """
    Main entry point for the application when run from the command line.
    """

    # Timing instance
    timing = Timing(['Snippet', 'Time [m]', 'Time [s]'])

    with timing.timeit_context_add('Pre-processing'):

        # Setup parse options command line
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parser = args.setup_parser(current_path + '/args/mfbn.json')
        options = parser.parse_args()
        args.update_json(options)
        args.check_output(options)

        if options.input and options.vertices is None:
            print('Vertices are required when input is given.')
            sys.exit(1)

    # Load bipartite graph
    with timing.timeit_context_add('Load graph'):

        source_graph = MGraph()
        if 'embedding_similarity' in options.similarity:
            source_graph.load(
                options.input,
                options.vertices,
                options.similarity,
                options.word_embs,
                options.doc_embs,
                options.y_filename,
                options.data_split
            )
        else:
            source_graph.load(options.input, options.vertices, options.similarity)

    # Coarsening
    with timing.timeit_context_add('Coarsening'):

        kwargs = dict(
            reduction_factor=options.reduction_factor, max_levels=options.max_levels,
            matching=options.matching, similarity=options.similarity, itr=options.itr,
            upper_bound=options.upper_bound, gmv=options.gmv,
            tolerance=options.tolerance, reverse=options.reverse, seed_priority=options.seed_priority,
            threads=options.threads
        )

        coarsening = Coarsening(source_graph, **kwargs)
        coarsening.run()

    # Save
    with timing.timeit_context_add('Save'):

        output = options.output
        for index, obj in enumerate(zip(coarsening.hierarchy_levels, coarsening.hierarchy_graphs)):
            level, coarsened_graph = obj
            index += 1

            if options.save_conf or options.show_conf:
                d = {
                    'source_input': options.input
                    , 'source_vertices': source_graph['vertices']
                    , 'source_vcount': source_graph.vcount()
                    , 'source_ecount': source_graph.ecount()
                    , 'coarsened_ecount': coarsened_graph.ecount()
                    , 'coarsened_vcount': coarsened_graph.vcount()
                    , 'coarsened_vertices': coarsened_graph['vertices']
                    , 'achieved_levels': coarsened_graph['level']
                    , 'reduction_factor': options.reduction_factor
                    , 'max_levels': options.max_levels
                    , 'similarity': options.similarity
                    , 'matching': options.matching
                    , 'upper_bound': options.upper_bound
                    , 'gmv': options.gmv
                    , 'itr': options.itr
                    , 'level': level
                }

            if options.save_conf:
                with open(output + '-' + str(index) + '-info.json', 'w+') as f:
                    json.dump(d, f, indent=4)

            if options.show_conf:
                print(json.dumps(d, indent=4))

            if options.save_ncol:
                coarsened_graph.write(output + '-' + str(index) + '.ncol', format='ncol')

            if options.save_source:
                with open(output + '-' + str(index) + '.source', 'w+') as f:
                    for v in coarsened_graph.vs():
                        f.write(' '.join(map(str, v['source'])) + '\n')

            if options.save_membership:
                membership = [0] * (source_graph['vertices'][0] + source_graph['vertices'][1])
                for v in coarsened_graph.vs():
                    for source in v['source']:
                        membership[source] = v.index
                numpy.savetxt(output + '-' + str(index) + '.membership', membership, fmt='%d')

            if options.save_predecessor:
                with open(output + '-' + str(index) + '.predecessor', 'w+') as f:
                    for v in coarsened_graph.vs():
                        f.write(' '.join(map(str, v['predecessor'])) + '\n')

            if options.save_successor:
                numpy.savetxt(output + '-' + str(index) + '.successor', coarsened_graph.vs['successor'], fmt='%d')

            if options.save_weight:
                numpy.savetxt(output + '-' + str(index) + '.weight', coarsened_graph.vs['weight'], fmt='%d')

            if options.save_gml:
                del coarsened_graph['adjlist']
                del coarsened_graph['similarity']
                coarsened_graph['layers'] = str(coarsened_graph['layers'])
                coarsened_graph['vertices'] = ','.join(map(str, coarsened_graph['vertices']))
                coarsened_graph['level'] = ','.join(map(str, coarsened_graph['level']))
                coarsened_graph.vs['name'] = map(str, range(0, coarsened_graph.vcount()))
                coarsened_graph.vs['type'] = map(str, coarsened_graph.vs['type'])
                coarsened_graph.vs['weight'] = map(str, coarsened_graph.vs['weight'])
                coarsened_graph.vs['successor'] = map(str, coarsened_graph.vs['successor'])
                for v in coarsened_graph.vs():
                    v['source'] = ','.join(map(str, v['source']))
                    v['predecessor'] = ','.join(map(str, v['predecessor']))
                coarsened_graph.write(output + '-' + str(index) + '.gml', format='gml')

            if not options.save_hierarchy:
                break

    if options.show_timing:
        timing.print_tabular()
    if options.save_timing_csv:
        timing.save_csv(output + '-timing.csv')
    if options.save_timing_json:
        timing.save_json(output + '-timing.json')


if __name__ == "__main__":
    sys.exit(main())
