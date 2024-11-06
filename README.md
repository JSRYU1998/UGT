# A Neural Operator Unifying Graph Neural Networks and Point-Transformer

Code for A Neural Operator Unifying Graph Neural Networks and Point-Transformer, submitted at IEEE ACCESS.

## Explain
Neural operators have emerged as a powerful tool for learning mappings between function spaces, particularly for solving partial differential equations.
This paper introduces a novel framework that unifies Graph Neural Networks and Transformers, combining their complementary strengths to enhance the approximation of operators derived from solutions of PDEs.
By integrating the feature extraction capabilities of GNNs with the attention mechanism of transformers, this approach efficiently captures information across multiple scales, from local patterns to global structures, thereby enhancing both the accuracy and efficiency of operator learning.

## Codes
Our code is implemented by torch and torch-geometric.
We referred to the following resources when implementing our code:

- [Fourier Neural Operator GitHub Repository](https://github.com/khassibi/fourier-neural-operator/tree/main)
- [PyTorch Geometric Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- [Simulating Complex Physics with Graph Networks (Medium Article)](https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05)
