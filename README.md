# SDF-based Graph Convolutional Networks

![Overview](https://github.com/hogunkee/SDF-based-Graph-Convolutional-Q-Networks/blob/main/figures/figure2_overview.png)

**Figure 1. The overview of SDF-GCQN.**

SDFGCN consists of two parts: a scene graph generator and a scene graph encoder.

![Overview](https://github.com/hogunkee/SDF-based-Graph-Convolutional-Q-Networks/blob/main/figures/figure3_graph_generator.png)

**Figure 2. Graph Generator.**

The scene graph consists of merging two scene subgraphs. The subgraph of each scene is a complete graph using the SDFs of objects as nodes.

![Overview](https://github.com/hogunkee/SDF-based-Graph-Convolutional-Q-Networks/blob/main/figures/figure4_graph_encoder.png)

**Figure 3. Graph Encoder.**

The graph encoder consists of CNN layer-based graph convolution layers. Since our scene graph has a node feature in the form of image, the network propagates the features through CNN blocks.

