#!/bin/bash

# cifar10
python ecsnn.py -act=ann -arch=vgg5 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=. -dataset=cifar10 -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=. -dataset=cifar10 -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=. -dataset=cifar10 -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=. -dataset=cifar10 -device=cpu -infer -split -split_dir=./splitted/

# caltech
python ecsnn.py -act=ann -arch=vgg5 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=. -dataset=caltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=. -dataset=caltech -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=. -dataset=caltech -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=. -dataset=caltech -device=cpu -infer -split -split_dir=./splitted/

# ncaltech
python ecsnn.py -act=ann -arch=vgg5 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./NCaltech/ -dataset=ncaltech -device=cpu -infer -split -split_dir=./splitted/

# cifar10_dvs
python ecsnn.py -act=ann -arch=vgg5 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./cifar10_dvs/ -dataset=cifar10_dvs -device=cpu -infer -split -split_dir=./splitted/

# GTZAN
python ecsnn.py -act=ann -arch=vgg5 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./GTZAN/ -dataset=gtzan -device=cpu -infer -split -split_dir=./splitted/

# UrbanSound
python ecsnn.py -act=ann -arch=vgg5 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg9 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=ann -arch=vgg16 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer 
python ecsnn.py -act=snn -arch=vgg5 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg9 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer -split -split_dir=./splitted/
python ecsnn.py -act=snn -arch=vgg16 -data_dir=./UrbanSound/ -dataset=urbansound -device=cpu -infer -split -split_dir=./splitted/