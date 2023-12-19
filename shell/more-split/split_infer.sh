#!/bin/bash

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=caltech
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=caltech
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=caltech

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10_dvs
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10_dvs
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=cifar10_dvs

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=ncaltech
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=ncaltech
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=ncaltech

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=gtzan
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=gtzan
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=gtzan

python ecsnn.py -arch=vgg5 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=urbansound
python ecsnn.py -arch=vgg9 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=urbansound
python ecsnn.py -arch=vgg16 -act=snn -device=cpu -infer -split -split_dir=./splitted_3/ -dataset=urbansound

