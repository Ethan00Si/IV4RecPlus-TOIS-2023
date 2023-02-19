# Enhancing Recommendation with Search Data in a Causal Learning Manner
This is the official implementation of the paper "**Enhancing Recommendation with Search Data in a Causal Learning Manner**" based on PyTorch.

This paper was accepted by TOIS. [[PDF](https://dl.acm.org/doi/10.1145/3582425)]

## Overview
We have applied the framework IV4Rec+ over several baselines, i.e., DIN, NRHUB, and SRGNN. The implementation can be found in folder `models`. IV4Rec+ is a model-agnostic framework, which can take existing sequential recommendation models as the underlying models. With two variations, denoted as IV4Rec+(UI) and IV4Rec+(I), IV4Rec+ has six instances over the three baselines mentioned above in our experiments. The instances of IV4Rec+(UI) are referred to as 'IV4Rec_UI_NRHUB', 'IV4Rec_UI_DIN', and 'IV4Rec_UI_SRGNN' in our codes. Similarly, the instances of IV4Rec+(I) are referred to as 'IV4Rec_I_NRHUB', 'IV4Rec_I_DIN', and 'IV4Rec_I_SRGNN'.


## Experimental Setting
All the hyper-parameter settings of IV4Rec+ instances and baselines on the MIND dataset can be found in the folder `config`. The settings of data can be found in file `config/const.py`.

## Dataset
Since the commercial datasets, i.e., Kuaishou-small and Kuaishou-large, are proprietary industrial datasets, here we release the experimental settings and data-processing codes of the MIND dataset. 


## Train and evaluate models:
Run codes in command line:
```bash
python3 main.py --name IV4Rec_I_DIN --dataset_name mind --gpu_id 0  --epochs 25 --tb True --model IV4Rec_I_DIN
python3 main.py --name IV4Rec_UI_DIN --dataset_name mind --gpu_id 0  --epochs 30 --tb True --model IV4Rec_UI_DIN
python3 main.py --name IV4Rec_I_NRHUB --dataset_name mind --gpu_id 0  --epochs 30 --tb True --model IV4Rec_I_NRHUB
python3 main.py --name IV4Rec_UI_NRHUB --dataset_name mind --gpu_id 0  --epochs 25 --tb True --model IV4Rec_UI_NRHUB
python3 main.py --name IV4Rec_I_SRGNN --dataset_name mind --gpu_id 0  --epochs 30 --tb True --model IV4Rec_I_SRGNN
python3 main.py --name IV4Rec_UI_SRGNN --dataset_name mind --gpu_id 0  --epochs 20 --tb True --model IV4Rec_UI_SRGNN
```

### Environments
The following python packages are required:
```
python==3.8.11
torch==1.9.0
numpy==1.21.2
pandas==1.3.3
scikit-learn==0.24.2
tqdm==4.62.2
PyYAML==6.0
```

The experiments are based on the following environments:
* CUDA Version: 11.1
* OS: CentOS Linux release 7.4.1708 (Core)
* GPU: The NVIDIA® T4 GPU
* CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz

## Citation
If you find our code or idea useful for your research, please cite our work.
```
@article{IV4Rec_plus_si_TOIS_2023,
author = {Si, Zihua and Sun, Zhongxiang and Zhang, Xiao and Xu, Jun and Song, Yang and Zang, Xiaoxue and Wen, Ji-Rong},
title = {Enhancing Recommendation with Search Data in a Causal Learning Manner},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3582425},
doi = {10.1145/3582425},
journal = {ACM Trans. Inf. Syst.},
month = {feb},
}
````

## Connect
If you have any questions, feel free to contact us through email zihua_si@ruc.edu.cn or GitHub issues. Thanks!