#!/bin/bash

# change split_dir: splitted_3, splitted_5, splitted_7, splitted_9

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10 -data_dir=. -b=128
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10 -data_dir=. -b=128
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10 -data_dir=. -b=128

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=caltech -data_dir=. -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=caltech -data_dir=. -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=caltech -data_dir=. -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10_dvs -data_dir=./cifar10_dvs/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10_dvs -data_dir=./cifar10_dvs/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=cifar10_dvs -data_dir=./cifar10_dvs/ -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=ncaltech -data_dir=./NCaltech/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=ncaltech -data_dir=./NCaltech/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=ncaltech -data_dir=./NCaltech/ -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=gtzan -data_dir=./GTZAN/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=gtzan -data_dir=./GTZAN/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=gtzan -data_dir=./GTZAN/ -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=urbansound -data_dir=./UrbanSound/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=urbansound -data_dir=./UrbanSound/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -split -split_dir=./splitted_9/ -dataset=urbansound -data_dir=./UrbanSound/ -b=16

