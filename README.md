# About DeepPQK and DeepQK

## DeepPQK_QK

DeepQPK and DeepPK is a deep learning architectures, which were developed for predicting protein-ligand binding affinity.

## Requirements:

PyTorch 1.10.0

Python 3.8

cuda 11.3

esm-2b


The easiest way to install the required packages:

```
pip install fair-esm
```

## **Model selection:**

```
from model3 import DeepDTAF # Includes pocket modules

model2.load_state_dict(torch.load('best_model.pt'))  # Includes pocket modules


from model_split_poc import DeepDTAF  # No pocket modules

model2.load_state_dict(torch.load('best_model_split.pt'))  # No pocket modules
```

## Affinity prediction：

In the model_run.py file

```
seq_mut="xxx"

pkt_list="xxx"

smi="xxx"
```

Run in a terminal

```
python model_run.py
```

