## Getting Started

### Prerequisites

To execute this code you have to create the right virtual environment.

```
conda env create -f environment.yml
```
### Run
Activate environment
```
conda activate qvt3
```
To run the QViT
```
 python main.py   --model qvt   --dataset eurosat   --num_channels 3   --epochs 10   --batch_size 32   --embed_dim 8   --hidden_dim 64   --patch_size 8   --classes 0 1 2
```
# EuroSAT Dataset Categories

The EuroSAT dataset is a land use and land cover classification dataset based on Sentinel-2 satellite images.  
It contains 10 classes representing different types of land cover. Each class is labeled with an integer ID from 0 to 9.

## 📂 Category Labels

| 类别编号 | 类别名称             |
| -------- | -------------------- |
| 0        | AnnualCrop           |
| 1        | Forest               |
| 2        | HerbaceousVegetation |
| 3        | Highway              |
| 4        | Industrial           |
| 5        | Pasture              |
| 6        | PermanentCrop        |
| 7        | Residential          |
| 8        | River                |
| 9        | SeaLake              |

These labels are used in training and evaluation when selecting specific classes using the `--classes` argument in your training scripts.

## 🧠 Example Usage

To train a model using only the first 3 classes:

```bash
python main.py --classes 0 1 2
```

To use all 10 classes:

```bash
python main.py --classes 0 1 2 3 4 5 6 7 8 9
```

---

*Data Source: EuroSAT (https://github.com/phelber/eurosat)*

