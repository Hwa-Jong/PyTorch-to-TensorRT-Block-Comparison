# Comparative Analysis of TensorRT Inference Speeds Across Various Neural Network Block Architectures

## Overview
This repository contains the implementation and results based on the research paper **"Comparative Analysis of TensorRT Inference Speeds Across Various Neural Network Block Architectures"** by Hwa-Jong Park. The paper systematically analyzes the inference speeds of neural network blocks across various combinations of normalization layers, activation functions, and input resolutions.

## Key Content
### Block Structure
<div style="text-align: center;">
    <img src="image/block_architecture.png" alt="Block Architecture" width="800">
</div>

### Experimental Results
#### 64×64 Resolution Results
<div style="text-align: center;">
    <img src="image/table_results_64.png" alt="Table 64x64 Results" width="800">
    <img src="image/results_64.png" alt="64x64 Results" width="800">
</div>

#### 256×256 Resolution Results
<div style="text-align: center;">
    <img src="image/table_results_256.png" alt="Table 256x256 Results" width="800">
    <img src="image/results_256.png" alt="256x256 Results" width="800">
</div>

#### 1024×1024 Resolution Results
<div style="text-align: center;">
    <img src="image/table_results_1024.png" alt="Table 1024x1024 Results" width="800">
    <img src="image/results_1024.png" alt="1024x1024 Results" width="800">
</div>

## Limitations
- Experiments were conducted using dummy images rather than real datasets.
- Testing was performed only on a single hardware configuration (NVIDIA GeForce RTX 3090 Ti), which limits the generalizability of the results.
- The analysis focused solely on inference speed, excluding memory usage, energy efficiency, and actual accuracy metrics.
