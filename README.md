## Usage
### Requirements
All experiments were conducted on a NVIDIA A6000 GPU. The initial learning rate was set to 0.001, and AdamW was used as the optimizer. 

```
python 3.8.19
torch 2.3.0+cu121
torchvision 0.18.0
numpy 1.22.1
pandas 2.0.3
pillow 10.3.0
Pkuseg 0.0.25
BERT ‘bert-base-chinese’ (HuggingFace)
```

## Training & Testing
```
python train.py
python test.py
```
