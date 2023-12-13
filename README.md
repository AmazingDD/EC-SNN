# EC-SNN
This is the repository of our article published in IJCAI 2024 "EC-SNN: Splitting Spiking Neural Network for Smart Sensors in Edge Computing" and of several follow-up studies.

## TODO List

- [x] 1213加急实验
- [ ] requirements章节的python包依赖补充好 @wentao
- [ ] how to run完善 @yudi
- [ ] appendix for all results
- [ ] there are some code redundancy in `ecsnn.py` (冗余但不影响运行效率，有机会再改吧)

## Overview

## Requirements

```

```

## How to run

```
python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -train 
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -train 
python ecsnn.py -arch=cifarnet -act=snn -prune -b=128 -split_dir=./splitted/ -device=cuda:4 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=cifarnet -act=ann -prune -b=128 -split_dir=./splitted/ -device=cuda:4 -apoz=56 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=cifarnet -act=snn -fusion -split_dir=./splitted/ -device=cuda:4 -b=128
python ecsnn.py -arch=cifarnet -act=ann -fusion -split_dir=./splitted/ -device=cuda:4 -b=128

python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -infer 
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -infer 

python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -infer -split -split_dir=./splitted/
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -infer -split -split_dir=./splitted/

python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -energy -device=cuda:4 -b=128
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -energy -device=cuda:4 -b=128

python ecsnn.py -arch=cifarnet -act=snn -device=cuda:4 -energy -split -split_dir=./splitted/ -device=cuda:4 -b=128
python ecsnn.py -arch=cifarnet -act=ann -device=cuda:4 -energy -split -split_dir=./splitted/ -device=cuda:4 -b=128
```

## Datasets

You can download experiment data and put them into the data folder. All data are available in the links below:

## Cite

Please cite the following paper if you find our work contributes to yours in any way:

```
@inproceedings{sun2020are,
  title={EC-SNN: Splitting Spiking Neural Network for Smart Sensors in Edge Computing},
  author={},
  booktitle={},
  year={2024}
}
```
