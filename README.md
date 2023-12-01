# 2023-ICML-DPL
This is the pytorch implementation of the [paper](https://proceedings.mlr.press/v202/song23c/song23c.pdf).

<img src="https://github.com/Vill-Lab/2023-ICML-DPL/blob/main/imgs/DPL_pipeline.png" width="527">

<img src="https://github.com/Vill-Lab/2023-ICML-DPL/blob/main/imgs/DPL_algorithm.png" width="527">

## Getting start
```
python main.py [--dataset DATASET] [--data_dir DATA_DIR] 
               [--net NET] [--batch_size BATCH_SIZE] [--gpu GPU] 
               [--lr LR] [--epoch EPOCH] [--resume] 
               [--alpha ALPHA] [--G_size G_SIZE] [--varepsilon VAREPSILON] [--rep_aug rep_aug]
```
- batch size: 128
- learning rate: 0.1
- training epoch: 200
- the hyperparameter $\alpha$: 0.005
- the size of *Guide Set*: 5% of the size of training set
- the hyperparameter $\varepsilon$: 0.04
- the approaches of replacement and augmentation: augmentation

## Citation
```
@inproceedings{song2023deep,
  title={Deep perturbation learning: enhancing the network performance via image perturbations},
  author={Song, Zifan and Gong, Xiao and Hu, Guosheng and Zhao, Cairong},
  booktitle={International Conference on Machine Learning},
  pages={32273--32287},
  year={2023},
  organization={PMLR}
}
```
