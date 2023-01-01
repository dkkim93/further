# FURTHER
Source code for ["Influencing Long-Term Behavior in Multiagent Reinforcement Learning"](https://arxiv.org/pdf/2203.03535.pdf) (NeurIPS 2022)

## Dependency
Known dependencies are (please refer to `requirements.txt`):
```
python 3.8.10
pip3.8
virtualenv 20.13.0
numpy==1.22.3
torch==1.10.1+cu111
gym==0.12.5
tensorboardX==1.2
pyyaml==3.12
gitpython==3.0.8
protobuf==3.20.0
```

## Setup
To avoid any conflict, please install virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
```
pip3.8 install --upgrade virtualenv
```
Please note that all the required dependencies will be automatically installed in the virtual environment by running the training script (`run.sh`).

## Run
First, please change the config argument in `run.sh`:
```
ibs.yaml (for IBS)
ic.yaml (for IC)
imp.yaml (for IMP)
```

After changing the config argument, start training by:
```
./run.sh
```

Lastly, to see the tensorboard logging during training:
```
tensorboard --logdir=logs
```

## Reference
```
@inproceedings{kim2022influencing,
title={Influencing Long-Term Behavior in Multiagent Reinforcement Learning},
author={Dong-Ki Kim and Matthew D Riemer and Miao Liu and Jakob Nicolaus Foerster and Michael Everett and Chuangchuang Sun and Gerald Tesauro and JONATHAN P HOW},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=_S9amb2-M-I}
}
```
