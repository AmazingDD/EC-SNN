# EC-SNN
This is the repository of our article published in IJCAI 2024 "EC-SNN: Splitting Spiking Neural Network for Smart Sensors in Edge Computing" and of several follow-up studies.

## TODO List

- [x] 1213加急实验
- [x] requirements章节的python包依赖补充好
- [ ] run /shell/infer.sh with raspberry @TWT
- [ ] run /shell/more-split/split_infer.sh 记得将split_dir换成3/5/7/9 @TWT
- [ ] 结果汇总到final页面中
- [ ] how to run完善
- [ ] appendix for all results
- [ ] there are some code redundancy in `ecsnn.py` (冗余但不影响运行效率，有机会再改吧)

## Overview

## Requirements

```
torch==2.0.1
torchvision==0.15.2
librosa==0.10.1
spikingjelly==0.0.0.0.14
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
opencv-python==4.8.1.78
```

## How to run

### model training

```
./shell/train.sh
./shell/ec-train.sh
```

### cifarnet quick start
```
# training
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -train 
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -train 
python ecsnn.py -arch=cifarnet -act=snn -prune -b=128 -split_dir=./splitted/ -device=cuda:4 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=cifarnet -act=ann -prune -b=128 -split_dir=./splitted/ -device=cuda:4 -apoz=56 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=cifarnet -act=snn -fusion -split_dir=./splitted/ -device=cuda:4 -b=128
python ecsnn.py -arch=cifarnet -act=ann -fusion -split_dir=./splitted/ -device=cuda:4 -b=128

# latency
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -infer 
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -infer 
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -infer -split -split_dir=./splitted/
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -infer -split -split_dir=./splitted/

# energy consumption
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -energy -b=128
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -energy -b=128
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -energy -split -split_dir=./splitted/ -b=128
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -energy -split -split_dir=./splitted/ -b=128
```

## Datasets

You can download experiment data and put them into the data folder. All data are available in the links below:

 - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02)
 - [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)
 - [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101) (Recommend One Drive)
 - [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
 - [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

## Cite

Please cite the following paper if you find our work contributes to yours in any way:

```
@inproceedings{,
  title={EC-SNN: Splitting Spiking Neural Network for Smart Sensors in Edge Computing},
  author={},
  booktitle={},
  year={2024}
}
```
