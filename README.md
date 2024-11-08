This repository is the official PyTorch implementation of "Semi-Supervised Coarsening of Bipartite Graphs for Text Classification via Graph Neural Network" published in the 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA).

## Requirements

This code was implemented using Python 3.11.5, CUDA 12.2 and the following packages:

- `networkx==3.3`
- `nltk==3.8.1`
- `numpy==1.26.4`
- `python_igraph==0.11.5`
- `PyYAML==6.0.1`
- `scikit_learn==1.4.2`
- `scipy==1.13.1`
- `torch==2.1.2`
- `torch_geometric==2.5.3`
- `python_igraph==0.11.5`

## How to run the code
In order to run our method, you must perform the steps described below.

Under construction.

The following arguments allow the modification of the GNN's hyperparameters:

- `--lr`

    Modifies the model's learning rate.
  
    Default: `1e-3`

- `--batch_size`

    Controls our method's batch size.
  
    Default: `32`

- `--hidden_dim`

    Number of dimensions used on GraphSAGE. 

    Default: `256`

- `--n_epochs`

    Number of training epochs.

    Default: `200`

- `--patience`

    Number of epochs without validation loss before early stopping.

    Default: `10`

- `--epoch_log`

    Prints information about the network's training every <epoch_log> steps.

    Default: `10`

- `--gpu`

    Trains the network using a GPU (if available).

    Default: `true`

- `--cpu`

    Trains the network using the CPU.

    Default: `false`

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
