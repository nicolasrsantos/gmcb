This repository is the official PyTorch implementation of "Semi-Supervised Coarsening of Bipartite Graphs for Text Classification via Graph Neural Network" published in the 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA).

## Requirements

This code was implemented using Python 3.11.2, CUDA 11.8 and the following packages:

- `pytorch==2.0.1`
- `torch-geometric==2.3.0`
- `numpy==1.24.1`
- `networkx==3.0`
- `scikit-learn=1.2.2`
- `igraph==0.10.4`
- `scipy==1.7.2`
- `pyyaml==6.0`

## How to run the code

## Reference

```
@InProceedings{santos2024gmcb,
  author={dos Santos, Nícolas Roque and Minatel, Diego and Valejo, Alan Demétrius Baria and de Andrade Lopes, Alneu},
  booktitle={2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Semi-Supervised Coarsening of Bipartite Graphs for Text Classification via Graph Neural Network}, 
  year={2024},
  pages={1-10},
  doi={10.1109/DSAA61799.2024.10722822}
}

```

## Acknowledgements

The Multilevel framework for bipartite networks (MFBN) plays an important role in our method. We adapted its original implementation to coarsen a graph using the cosine similarity of the words present in the datasets we used. For more information on MFBN, you can check the following repository: https://github.com/alanvalejo/mfbn.
