#!/bin/bash

# origin energy
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=cifar10 -b=128 

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16 
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=. -dataset=caltech -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=32 

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16 
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=./NCaltech/ -dataset=ncaltech -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16 
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=./GTZAN/ -dataset=gtzan -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16 
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16 
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16 
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16 
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:2 -energy -data_dir=./UrbanSound/ -dataset=urbansound -b=16

# split energy device=1
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=. -dataset=cifar10 -energy -split -split_dir=./splitted/ -b=128
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=. -dataset=cifar10 -energy -split -split_dir=./splitted/ -b=128
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=. -dataset=cifar10 -energy -split -split_dir=./splitted/ -b=128
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=. -dataset=caltech -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=. -dataset=caltech -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=. -dataset=caltech -energy -split -split_dir=./splitted/ -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -energy -split -split_dir=./splitted/ -b=32
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -energy -split -split_dir=./splitted/ -b=32
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -energy -split -split_dir=./splitted/ -b=32
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=./NCaltech/ -dataset=ncaltech -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=./NCaltech/ -dataset=ncaltech -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=./NCaltech/ -dataset=ncaltech -energy -split -split_dir=./splitted/ -b=16

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=./GTZAN/ -dataset=gtzan -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=./GTZAN/ -dataset=gtzan -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=./GTZAN/ -dataset=gtzan -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -data_dir=./UrbanSound/ -dataset=urbansound -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -data_dir=./UrbanSound/ -dataset=urbansound -energy -split -split_dir=./splitted/ -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -data_dir=./UrbanSound/ -dataset=urbansound -energy -split -split_dir=./splitted/ -b=16
