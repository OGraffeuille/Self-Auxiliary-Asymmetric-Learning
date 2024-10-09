# Self-Auxiliary Asymmetric Learning
This repository contains the source code of Asymmetric Transfer in Multi-Task Learning with Self-Auxiliaries

This repository is based on the implementation of [Auto-Lambda](https://github.com/lorenmt/auto-lambda).

## Dependencies
This implementation was tested with:

* Python 3.8.5
* pytorch 1.12 with Cuda 11.6
* numpy 1.22.4

## Baselines
The following baselines are included:

### Weighting-based:
- **Equal** - All task weightings are 1. `--weight equal`
- **Uncertainty** - [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115) `--weight uncert`
- **Dynamic Weight Average** - [https://arxiv.org/abs/1803.10704](https://arxiv.org/abs/1803.10704) `--weight dwa`
- **Auto-Lambda** - [https://arxiv.org/pdf/2202.03091](https://arxiv.org/pdf/2202.03091) `--weight autol`

### Gradient-based:
- **GradDrop** -  [https://arxiv.org/abs/2010.06808](https://arxiv.org/abs/2010.06808) `--grad_method graddrop`
- **PCGrad** - [https://arxiv.org/abs/2001.06782](https://arxiv.org/abs/2001.06782) `--grad_method pcgrad`
- **CAGrad** - [https://arxiv.org/abs/2110.14048](https://arxiv.org/abs/2110.14048) `--grad_method cagrad`

## Datasets
The following datasets are available:

| DATASET_NAME  | TASK_NAME                                           |  
|---------------|-----------------------------------------------------|
| `nyuv2`       | ['seg', 'depth', 'normal']                          |
| `cityscapes`  | ['seg', 'part_seg', 'disp']                         |
| `celeba`      | ['class_0', 'class_1, ... 'class_8']                |
| `robotarm`    | ['regression_0', 'regression_1, ... 'regression_9'] |
| `global_chl`  | ['regression_0', 'regression_1, ... 'regression_9'] |

## Experiments

Single-Task Learning:
```
python trainer.py --dataset [DATASET_NAME] --task [TASK_NAME] --gpu 0 
```

Multi-Task Learning:
```
python trainer.py --dataset [DATASET_NAME] --network branch --task all --weight [equal, uncert, dwa, autol] --grad_method [graddrop, pcgrad, cagrad] --gpu 0  
```

SAAL_e:
```
python trainer.py --dataset [DATASET_NAME] --network branch --task all_aux_positive --gpu 0  
```

SAAL_w:
```
python trainer.py --dataset [DATASET_NAME] --network branch --task all_aux --weight autol --gpu 0  
```

SAAL_ew:
```
python trainer.py --dataset [DATASET_NAME] --network branch --task all_aux_positive --weight autol --gpu 0  
```

