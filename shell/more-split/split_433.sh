#!/bin/bash

# cifar10
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=128

python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=128

python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -prune -b=128 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=cifar10 -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=128

# caltech
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=. -dataset=caltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

# cifar10_dvs
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

# NCaltech
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./NCaltech/ -dataset=ncaltech -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

# GTZAN
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./GTZAN/ -dataset=gtzan -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

# UrbanSound
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg5 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg9 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16

python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 0 1 2 3 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 4 5 6 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -prune -b=16 -split_dir=./splitted_433/ -device=cuda:2 -apoz=95 -c 7 8 9 
python ecsnn.py -arch=vgg16 -act=snn -data_dir=./UrbanSound/ -dataset=urbansound -fusion -split_dir=./splitted_433/ -device=cuda:2 -b=16
