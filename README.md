## Getting Started

### Prerequisites

To execute this code you have to create the right virtual environment.

```
conda env create -f environment.yml
```
To run the QViT
```
 python main.py   --model qvt   --dataset eurosat   --num_channels 3   --epochs 10   --batch_size 32   --embed_dim 8   --hidden_dim 64   --patch_size 8   --classes 0 1 2
```
