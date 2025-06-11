# ComfyUI-Unload-Model

Forked from: https://github.com/SeanScripts/ComfyUI-Unload-Model

This fork add more control over node execution order.

## Installation

1. Clone this repo into the `custom_nodes` folder:
```
git clone https://github.com/kildom/ComfyUI-Unload-Model.git
```
2. Restart the ComfyUI server.

## Usage

Add the `ForceUnloadModels` to the workflow. When executed, it will unload all model except those that are passing through this node using inputs and outputs.

To ensure that all nodes that uses models to unload are already executed, pass their ouputs through this node.

To ensure that some model is not loaded before this node execution, pass some dummy parameter (e.g. model file name string) through this node.
