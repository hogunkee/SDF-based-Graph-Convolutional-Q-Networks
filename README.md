# SDF-based Graph Convolutional Networks

Torch implementation of SDF-GCQN.

![Overview](https://github.com/hogunkee/SDF-based-Graph-Convolutional-Q-Networks/blob/main/figures/Figure2_overview.png)

**Figure 1. The overview of SDF-GCQN.**

SDF-GCQN consists of two parts: a scene graph generator and a scene graph encoder.
The scene graph generation consists of merging two scene subgraphs. 
The subgraph of each scene is a complete graph using the SDFs of objects as nodes.
The scene graph encoder consists of CNN layer-based graph convolution layers.


## Usage
To train a model:

```
python dqn_train.py
```

To test the trained model:

```
python dqn_eval.py --model_path [MODEL_NAME]
```

To test the rule-based method:

```
python rulebased_eval.py
```

## License
MIT License.
