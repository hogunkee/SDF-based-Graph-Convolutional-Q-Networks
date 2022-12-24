# SDF-based Graph Convolutional Networks

![Overview](https://github.com/hogunkee/SDF-based-Graph-Convolutional-Q-Networks/blob/main/figures/Figure2_overview.png)

**Figure 1. The overview of SDF-GCQN.**

SDFGCN consists of two parts: a scene graph generator and a scene graph encoder.
The scene graph consists of merging two scene subgraphs. The subgraph of each scene is a complete graph using the SDFs of objects as nodes.
The graph encoder consists of CNN layer-based graph convolution layers. Since our scene graph has a node feature in the form of image, the network propagates the features through CNN blocks.

