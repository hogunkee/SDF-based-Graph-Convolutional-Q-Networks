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
# Rendering on #
python dqn_train.py --render --show_q

# Rendering off #
python dqn_train.py --gpu [gpu_idx] --show_q
```

To test the trained model:

```
# Rendering on #
python dqn_eval.py --model_path [MODEL_NAME] --render --show_q

# Rendering off #
python dqn_eval.py --model_path [MODEL_NAME] --gpu [gpu_idx] --show_q
```

To test the rule-based method:

```
# Rendering on #
python rulebased_eval.py --render --show_q

# Rendering off #
python rulebased_eval.py --gpu [gpu_idx] --show_q
```

## License
MIT License.

