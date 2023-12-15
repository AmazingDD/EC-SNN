#!/bin/bash

python ecsnn.py -arch=vgg5 -act=snn -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:5 -train -data_dir=. -dataset=cifar10 -b=128

python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:3 -train -data_dir=. -dataset=caltech -b=16

python ecsnn.py -arch=vgg16 -act=ann -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -train -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -b=16


python ecsnn.py -arch=vgg16 -act=ann -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -train -data_dir=./NCaltech/ -dataset=ncaltech -b=16

# gtzan urbansound 
python ecsnn.py -arch=vgg16 -act=ann -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -train -data_dir=./GTZAN/ -dataset=gtzan -b=16


python ecsnn.py -arch=vgg16 -act=ann -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
python ecsnn.py -arch=vgg9 -act=ann -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
python ecsnn.py -arch=vgg5 -act=ann -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
python ecsnn.py -arch=vgg16 -act=snn -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
python ecsnn.py -arch=vgg9 -act=snn -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
python ecsnn.py -arch=vgg5 -act=snn -device=cuda:3 -train -data_dir=./UrbanSound/ -dataset=urbansound -b=16
