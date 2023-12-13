import os
import cv2
import librosa
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

from spikingjelly.activation_based import neuron
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101 


dataset = ['mnist', 'cifar10', 'caltech', 'cifar10_dvs', 'nmnist', 'ncaltech']

img_size_ref = {
    'mnist': (28, 28),
    'nmnist': (34, 34),
    'cifar10': (32, 32),
    'cifar10_dvs': (128, 128),
    'caltech': (224, 224),
    'ncaltech': (224, 224),
    'gtzan': (96, 323),
    'urbansound': (40, 173), 
}

input_dim_ref = {
    'mnist' : 1,
    'nmnist': 2,
    'cifar10': 3,
    'cifar10_dvs': 2,
    'caltech': 3,
    'ncaltech': 2,
    'gtzan': 1,
    'urbansound': 1,
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

class Logger:
    ''' spikingjelly induce logging not work '''
    def __init__(self, args, state=None, desc=None):
        log_root = args.out_dir
        dir_name = os.path.dirname(log_root)
        ensure_dir(dir_name)

        out_dir = os.path.join(log_root, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_{args.epochs}epochs')

        ensure_dir(out_dir)

        logfilename = f'{desc}_record.log' # _{get_local_time()}
        logfilepath = os.path.join(out_dir, logfilename)

        self.filename = logfilepath

        f = open(logfilepath, 'w', encoding='utf-8')
        f.write(str(args) + '\n')
        f.flush()
        f.close()


    def info(self, s=None):
        print(s)
        f = open(self.filename, 'a', encoding='utf-8')
        f.write(f'[{get_local_time()}] - {s}\n')
        f.flush()
        f.close()



def generate_infer_one_sample(T=5):
    print('Load infer data')
    if not os.path.exists('./infer_data/'):
        os.makedirs('./infer_data/')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='.',
        train=False,
        transform=transform_test,
        download=True)
    frame, _ = next(iter(test_dataset))
    print('mnist: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/mnist_frame.pt')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='.',
        train=False,
        transform=transform_test,
        download=True)  
    frame, _ = next(iter(test_dataset))
    print('cifar10: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/cifar10_frame.pt')

    transform_all = transforms.Compose([
        # transforms.ToPILImage(),  # already PILImage
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_dataset = torchvision.datasets.Caltech101(
        root='.',
        transform=transform_all,
        download=True)

    frame, _ = next(iter(test_dataset))
    print('caltech: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/caltech_frame.pt')

    test_dataset = CIFAR10DVS(
        root='./cifar10_dvs', 
        data_type='frame', 
        frames_number=T, 
        split_by='number')

    frame, _ = next(iter(test_dataset))
    frame = torch.tensor(frame)
    print('cifar10-dvs: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/cifar10_dvs_frame.pt')

    test_dataset = NMNIST(
        root='./NMNIST',
        train=False,
        data_type='frame',
        frames_number=T,
        split_by='number')

    frame, _ = next(iter(test_dataset))
    frame = torch.tensor(frame)
    print('n-mnist: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/nmnist_frame.pt')

    transform_all = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Resize((224, 224), antialias=True),
    ])

    test_dataset = NCaltech101(
        root='./NCaltech', 
        data_type='frame',
        frames_number=T,
        transform=transform_all,
        split_by='number')

    frame, _ = next(iter(test_dataset))
    print('n-caltech: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/ncaltech_frame.pt')

# select top10 from 101
class CaltechTop10(torch.utils.data.Dataset):
    def __init__(self, data):
        top10 = [5, 3, 0, 1, 94, 2, 12, 19, 55, 23]
        ref = {top10[i] : i for i in range(10)}
        self.data = []
        for x, y in data:
            if y in top10:
                self.data.append((x, ref[y]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tar = self.data[idx]
        return tar[0], tar[1]
    
def split_retrain(x_train, x_test, y_train, y_test, target_class, num_cls=10):
    ''' resample dataset for specific classes for pruning '''
    mapping = {k: v for k, v in zip(target_class, range(1, len(target_class) + 1))}

    length = int(len(y_train) / (num_cls * 10) - len(target_class))

    x_retrain_train = torch.full([0] + [dim for dim in x_train.shape[1:]], fill_value=0.)
    y_retrain_train = torch.full([0], fill_value=0, dtype=torch.uint8)
    x_retrain_test = torch.full([0] + [dim for dim in x_train.shape[1:]], fill_value=0.)
    y_retrain_test = torch.full([0], fill_value=0, dtype=torch.uint8)


    for i in range(num_cls):
        if i in target_class:
            new_label = mapping[i]

            positive_image = x_train[y_train == i]
            x_retrain_train = torch.concat([x_retrain_train, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_train[y_train == i]), dtype=torch.uint8)
            y_retrain_train = torch.concat([y_retrain_train, positive_label], dim=0)
            
            positive_image = x_test[y_test == i]
            x_retrain_test = torch.concat([x_retrain_test, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_test[y_test == i]), dtype=torch.uint8)
            y_retrain_test = torch.concat([y_retrain_test, positive_label], dim=0)

        else:
            negative_image = x_train[y_train == i]
            temp_l = len(negative_image) if len(negative_image) <= length else length
            rnd_idx = torch.randperm(len(y_train[y_train == i]))[:temp_l]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros([temp_l], dtype=torch.uint8)
            x_retrain_train = torch.concat([x_retrain_train, negative_image], dim=0)
            y_retrain_train = torch.concat([y_retrain_train, negative_label], dim=0)

            negative_image = x_test[y_test == i]
            temp_l = len(negative_image) if len(negative_image) <= length else length
            rnd_idx = torch.randperm(len(y_test[y_test == i]))[:temp_l]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros([temp_l], dtype=torch.uint8)
            x_retrain_test = torch.concat ([x_retrain_test, negative_image], dim=0)
            y_retrain_test = torch.concat([y_retrain_test, negative_label], dim=0)
    
    return x_retrain_train, y_retrain_train, x_retrain_test, y_retrain_test

def spike_count(layers, data):
    cnt = 0
    inputs = data
    for layer in layers:
        inputs = layer(inputs)
        if isinstance(layer, (neuron.LIFNode, nn.ReLU)):
            cnt += inputs.count_nonzero()

    return cnt, inputs

# audio GTZAN
def get_gtzan_dataset(path='./GTZAN/'):
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    num_div=8

    path = f'{path}genres_original'

    music_dataset = []
    genre_target = []
    for root, _, files in os.walk(path):
        for name in files:
            filename = os.path.join(root, name)
            if filename != f'{path}/jazz/jazz.00054.wav':
                music_dataset.append(filename)
                genre_target.append(filename.split("/")[3])

    mel_spec=[]
    genre_new=[]
    for idx, wav in enumerate(music_dataset):
        y, sfr = librosa.load(wav)
        div= np.split(y[:660000], num_div) # different length, preserve the first 660000 features
        for chunck in div:
            melSpec = librosa.feature.melspectrogram(y=chunck, sr=sfr, n_mels=N_MELS,hop_length=HOP_LEN, n_fft=N_FFT)
            melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
            mel_spec.append(melSpec_dB)
            genre_new.append(genre_target[idx])

    labels = os.listdir(f'{path}')
    # idx2labels = {k: v for k, v in enumerate(labels)}
    labels2idx = {v: k for k, v in enumerate(labels)}   

    genre_id = [labels2idx[item] for item in genre_new]

    X_train, X_test, y_train, y_test = train_test_split(mel_spec, genre_id, test_size=0.2)
    X_train, X_test = np.array(X_train), np.array(X_test)

    # unsqueeze to create channel dimension
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset

# audio urbansound
def get_urbansound_dataset(path='./UrbanSound/'):
    up_height = 40
    up_width = 173

    X = []
    y = []

    df = pd.read_csv(os.path.join(path, 'UrbanSound8K.csv'))
    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f'Finish {idx + 1} files resample')
        file_name = row['slice_file_name']
        folder_num = row['fold']
        label = row['classID']

        fp = os.path.join(path, f'fold{folder_num}', file_name)
        raw , sr = librosa.load(fp, res_type='kaiser_fast') # pip install resampy
        X_ = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=40)
        up_points = (up_width, up_height)
        X_ = cv2.resize(X_, up_points, interpolation=cv2.INTER_LINEAR)
        X.append(X_)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2)

    # create channel
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset

    