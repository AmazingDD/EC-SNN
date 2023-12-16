#!/bin/bash

# static
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9

python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9

# N-data
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9 # 99

python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=95 -c 0 1 2 3 4 5 6 7 8 9


# audio
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9

python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9 # 97
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9 # 97
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted/ -device=cuda:3 -apoz=93 -c 0 1 2 3 4 5 6 7 8 9 # 97


# fusion
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted/ -device=cuda:2 -b=128
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted/ -device=cuda:2 -b=128
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted/ -device=cuda:2 -b=128

python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted/ -device=cuda:2 -b=16
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted/ -device=cuda:2 -b=16