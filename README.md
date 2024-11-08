This repository is the official PyTorch implementation of "Semi-Supervised Coarsening of Bipartite Graphs for Text Classification via Graph Neural Network" published in the 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA).

## Requirements

This code was implemented using Python 3.11.5, CUDA 12.2 and the following packages:

- `pytorch==2.1.2`
- `torch-geometric==2.5.3`
- `numpy==1.26.4`
- `networkx==3.1`
- `scikit-learn=1.4.2`
- `igraph==0.10.4`
- `scipy==1.13.0`
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
