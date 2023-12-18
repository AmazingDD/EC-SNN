import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.activation_based import functional, encoding, neuron, surrogate, layer

from model import VGG, Fusion
from utils import *


############## Reproducibility ##############
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################

parser = argparse.ArgumentParser(description='EC-SNN interface for all experiments')

parser.add_argument('-num_cls', default=10, type=int, help='number of class for classification')
parser.add_argument('-model_dir', type=str, default='./trained/', help='root dir for saving trained model')
parser.add_argument('-split_dir', type=str, default='./splitted/', help='root dir for saving trained model')
parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-device', default='cpu', help='device')  # cuda:0
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-dataset', default='cifar10', help='dataset name')
parser.add_argument('-arch', default='cifarnet', help='dataset name')
parser.add_argument('-act', default='snn', help='ANN or SNN, default is snn, determine relu or lif for spikes')

parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
parser.add_argument('-b', default=128, type=int, help='batch size')
parser.add_argument('-epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-tau', default=4./3, type=float, help='parameter tau of LIF neuron')

parser.add_argument('-encode', default='r', type=str, help='spike encoding methode. (p)Poisson, (l)Latency, or (r)Raw')
parser.add_argument('-train', action='store_true', help='generate origin trained model')
parser.add_argument('-prune', action='store_true', help='generate pruned trained model')
parser.add_argument('-fusion', action='store_true', help='fusion the final results with all networks in split_dir')
parser.add_argument('-infer', action='store_true', help='get infer time result')
parser.add_argument('-energy', action='store_true', help='get energy consumption result')
parser.add_argument('-split', action='store_true', help='split mode or single mode')


parser.add_argument('-apoz', '--apoz_threshold', type=float, default=93., help="APOZ threshold for filter's activation map")
parser.add_argument('-c', '--split_class', type=int, nargs='+', help='class code for splitting')
parser.add_argument('-min_f', type=int, default=16, help="minimym number of filters for specific conv-layer after split")

args = parser.parse_args()

desc = ''
if args.train:
    desc += 'train'
if args.prune:
    desc += 'prune'
if args.infer:
    desc += 'infer'
if args.fusion:
    desc += 'fusion'
if args.energy:
    desc += 'energy'
if args.split:
    assert (args.split and args.energy) or (args.split and args.infer), 'arg split must be paired with arg energy or infer'
    desc = 'split_' + desc

logger = Logger(args, desc=desc)

logger.info(args)
ensure_dir(f'{args.model_dir}')

logger.info('Load data')
if args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)
    test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)

elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform_train,
        download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform_test,
        download=True)  
elif args.dataset == 'caltech':
    if not os.path.exists('./caltech_dataset.pt'):
        # batch=16
        transform_all = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        dataset = torchvision.datasets.Caltech101(
            root=args.data_dir,
            transform=transform_all,
            download=True)
        dataset = CaltechTop10(dataset)
        train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
        torch.save([train_dataset, test_dataset], './caltech_dataset.pt')
    else:
        print('Files already processed in data_dir')
        train_dataset, test_dataset = torch.load('./caltech_dataset.pt')

elif args.dataset == 'cifar10_dvs':
    if not os.path.exists('./cifar10_dvs_dataset.pt'):
        dataset = CIFAR10DVS(
            root=args.data_dir, 
            data_type='frame', 
            frames_number=args.T, 
            split_by='number')
        train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
        torch.save([train_dataset, test_dataset], './cifar10_dvs_dataset.pt')
    else:
        print('Files already processed in data_dir')
        train_dataset, test_dataset = torch.load('./cifar10_dvs_dataset.pt')

elif args.dataset == 'nmnist':
    train_dataset = NMNIST(
        root=args.data_dir,
        train=True,
        data_type='frame',
        frames_number=args.T,
        split_by='number')
    test_dataset = NMNIST(
        root=args.data_dir,
        train=False,
        data_type='frame',
        frames_number=args.T,
        split_by='number')

elif args.dataset == 'ncaltech':
    if not os.path.exists('./ncaltech_dataset.pt'):
        transform_all = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x)),
            transforms.Resize((224, 224), antialias=True),
        ])

        dataset = NCaltech101(
            root=args.data_dir, 
            data_type='frame',
            frames_number=args.T,
            transform=transform_all,
            split_by='number')
        
        dataset = CaltechTop10(dataset)
        
        train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
        torch.save([train_dataset, test_dataset], './ncaltech_dataset.pt')
    else:
        print('Files already processed in data_dir')
        train_dataset, test_dataset = torch.load('./ncaltech_dataset.pt')

elif args.dataset == 'gtzan':
    if not os.path.exists('./gtzan_dataset.pt'):
        train_dataset, test_dataset = get_gtzan_dataset(args.data_dir) 
        torch.save([train_dataset, test_dataset], './gtzan_dataset.pt')
    else:
        print('Files already processed in data_dir')
        train_dataset, test_dataset = torch.load('./gtzan_dataset.pt')

elif args.dataset == 'urbansound':
    if not os.path.exists('./urbansound_dataset.pt'):
        train_dataset, test_dataset = get_urbansound_dataset(args.data_dir) 
        torch.save([train_dataset, test_dataset], './urbansound_dataset.pt')
    else:
        print('Files already processed in data_dir')
        train_dataset, test_dataset = torch.load('./urbansound_dataset.pt')

else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}...')

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.b,
    shuffle=True,
    drop_last=True, 
    num_workers=args.j, 
    pin_memory=True)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.b,
    shuffle=False,
    drop_last=False, 
    num_workers=args.j, 
    pin_memory=True)

logger.info(f'[{args.dataset}] train samples: {len(train_dataset)}, test samples: {len(test_dataset)}')

logger.info('Create new model')
net = VGG(
    act=args.act, 
    arch=args.arch, 
    num_cls=args.num_cls, 
    input_dim=input_dim_ref[args.dataset], 
    img_height=img_size_ref[args.dataset][0],
    img_width=img_size_ref[args.dataset][1],
    spiking_neuron=neuron.LIFNode, tau=args.tau, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)


logger.info('Network Architecture Details:')
logger.info('\n' + str(net))
logger.info('Arguments Settings:')
logger.info(str(args))
logger.info('Running Command:')
logger.info(' '.join(sys.argv))

encoder = None
if args.encode == 'p':
    encoder = encoding.PoissonEncoder()
elif args.encode == 'l':
    encoder = encoding.LatencyEncoder()
elif args.encode == 'r':
    pass
else:
    raise NotImplementedError(f'invalid encoding method: {args.encode}')

if args.train:
    net.to(args.device)

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    max_test_acc = -1
    train_time_record = []
    for epoch in range(args.epochs):
        logger.info(f'start epoch {epoch + 1}')
        net.train()

        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            out_fr = 0.
            loss = 0.

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):
                    output = net(img[t])
                    out_fr += output
                    loss += F.cross_entropy(output, label)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    output = net(encoded_img)
                    out_fr += output
                    loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time() - start_time
        train_loss /= train_samples
        train_acc /= train_samples

        train_time_record.append(train_time)

        lr_scheduler.step()

        net.eval()

        start_time = time.time()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        output = net(img[t])
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        output = net(encoded_img)
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_time = time.time() - start_time
        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth'))

        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        logger.info(f'train time: {train_time:.3f}s, test time: {test_time:.3f}s')

    logger.info('Metrics:')
    logger.info(f'Accuracy: {max_test_acc:.4f}\tTraining Elapse Time per epoch: {np.mean(train_time_record):.4f}s\t\n')


if args.prune:
    ensure_dir(os.path.join(args.split_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint'))
    
    checkpoint = torch.load(os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth'), map_location='cpu')
    net = checkpoint['net']
    logger.info(f'Load existing model')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        drop_last=True, 
        num_workers=args.j, 
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        drop_last=False, 
        num_workers=args.j, 
        pin_memory=True)
    
    # select target class
    target_class = sorted(args.split_class) 
    pmodel_name = ''.join([str(x) for x in target_class])

    p_num_cls = len(target_class) + 1

    x_train, y_train = next(iter(train_data_loader))
    x_test, y_test = next(iter(test_data_loader))
    del train_dataset, test_dataset

    x_retrain_train, y_retrain_train , x_retrain_test, y_retrain_test = split_retrain(
        x_train, x_test , y_train, y_test, target_class, args.num_cls)
    logger.info('Finish extracting pruning data from original data')

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CustomDataset(x_retrain_train, y_retrain_train)
    test_dataset = CustomDataset(x_retrain_test, y_retrain_test)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True, 
        num_workers=args.j, 
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False, 
        num_workers=args.j, 
        pin_memory=True)

    prune_sample = x_retrain_train[y_retrain_train!=0] # (N,C,H,W) for non-neu data, (N, T, C, H, W) for neu data
    logger.info(f'pruned samples number: {prune_sample.shape}')

    iter_len = args.b # set chunck size same as batch size for train, then it will always meet the memory limitation
    idx_set = list(range(0, len(prune_sample), iter_len))
    if idx_set[-1] + iter_len >= len(prune_sample):
        idx_set.append(len(prune_sample)) 

    activation = {}
    def get_activation(name):
        def hook(m, input, output):
            activation[name] = output.detach()
        return hook
    
    follow_conv = False
    num_conv = 0 
    i_h_dim = img_size_ref[args.dataset][0]
    i_w_dim = img_size_ref[args.dataset][1]
    area_conv_hw = []
    for n, l in net.conv_fc.named_children():
        if isinstance(l, layer.Conv2d):
            follow_conv = True
            hw = i_h_dim * i_w_dim
            area_conv_hw.append(hw)
        elif isinstance(l, (neuron.LIFNode, nn.ReLU)) and follow_conv is True:
            l.register_forward_hook(get_activation(f'conv{num_conv}'))
            num_conv += 1
            follow_conv = False
            print(f'conv with index {n} in nn.sequential is hooked')
        elif isinstance(l, (layer.AvgPool2d, layer.MaxPool2d)):
            i_h_dim //= 2
            i_w_dim //= 2
        else:
            pass
    logger.info(f'Finish add forward-hook to model, total conv layer number: {num_conv}') 

    areas = np.array(area_conv_hw)
    apozs = [0. for _ in range(num_conv)]

    net.to(args.device)

    p_cnt = 0
    for start_idx, end_idx in zip(idx_set[:-1], idx_set[1:]):
        p_prune_sample = prune_sample[start_idx:end_idx] # N, C, H, W / N, T, C, H, W
        p_prune_sample = p_prune_sample.to(args.device)

        nzs = [0. for _ in range(num_conv)]

        if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
            p_prune_sample = p_prune_sample.transpose(0, 1)
            for t in range(args.T):
                _ = net(p_prune_sample[t])
                for k in range(num_conv):
                    act = torch.permute(activation[f'conv{k}'], dims=[1,0,2,3]) # N,C,H,W -> C, N, H, W
                    act = act.reshape(act.shape[0], -1) # C, N, H, W -> C, NHW
                    nz = act.shape[1] - act.count_nonzero(dim=1) # find out how many zero in NHW 
                    nzs[k] += nz
        else:
            for t in range(args.T):
                encoded_img = encoder(p_prune_sample) if encoder is not None else p_prune_sample
                _ = net(encoded_img)
                for k in range(num_conv):
                    act = torch.permute(activation[f'conv{k}'], dims=[1,0,2,3]) # N,C,H,W -> C, N, H, W
                    act = act.reshape(act.shape[0], -1) # C, N, H, W -> C, NHW
                    nz = act.shape[1] - act.count_nonzero(dim=1) # find out how many zero in NHW 
                    nzs[k] += nz

        nzs = [nz / args.T for nz in nzs]  # non-zero number per conv per T
        p_areas = areas * (end_idx - start_idx) # Nhw
        
        for nc in range(num_conv):
            apoz = nzs[nc] / p_areas[nc]
            apozs[nc] += apoz
        
        print(f'Finish extracting apoz with pruned samples {start_idx}-{end_idx}')
        p_cnt += 1

        functional.reset_net(net)

    apozs = [(apoz / p_cnt).tolist() for apoz in apozs]

    # print(apozs)

    layer_index = []
    for nc in range(num_conv):
        apoz = apozs[nc]
        idxs = []
        for na in range(len(apoz)):
            A = apoz[na]
            if (A > ((args.apoz_threshold - nc) / 100)) & (args.min_f < len(apoz) - idxs.count(0)):
                idxs.append(0)
            else:
                idxs.append(1)
        layer_index.append(idxs)

    logger.info('Finish ranking filter importance')
    for i in range(len(layer_index)):
        ele = layer_index[i]
        logger.info(f'conv{i} shrink to {np.sum(ele)}')

    new_conv_fc = nn.Sequential()
    c_dim = input_dim_ref[args.dataset]
    i_h_dim = img_size_ref[args.dataset][0]
    i_w_dim = img_size_ref[args.dataset][1]
    bias_flag = True if args.act == 'ann' else False
    index = 0
    conv_idx_rec = []
    conv_idx = 0

    for l in net.conv_fc:
        if isinstance(l, layer.Conv2d):
            channels = np.array(layer_index[index])
            num_channels = channels.sum()
            new_conv_fc.append(
                layer.Conv2d(c_dim, num_channels, kernel_size=3, padding=1, bias=bias_flag))

            c_dim = num_channels
            index += 1
            conv_idx_rec.append(conv_idx)
        elif isinstance(l, layer.BatchNorm2d):
            new_conv_fc.append(layer.BatchNorm2d(c_dim)) 
        elif isinstance(l, (layer.AvgPool2d, layer.MaxPool2d)):
            new_conv_fc.append(
                layer.AvgPool2d(kernel_size=2) if args.act == 'snn' else layer.MaxPool2d(kernel_size=2))
            i_h_dim //= 2
            i_w_dim //= 2
        elif isinstance(l, layer.Dropout):
            new_conv_fc.append(layer.Dropout(0.5))
        elif isinstance(l, layer.Flatten):
            new_conv_fc.append(layer.Flatten())
        elif isinstance(l, nn.Linear):
            new_conv_fc.append(layer.Linear(i_h_dim * i_w_dim * c_dim, 384, bias=bias_flag))
            new_conv_fc.append(layer.Dropout(0.5))
            new_conv_fc.append(layer.Linear(384, 192, bias=bias_flag))
            new_conv_fc.append(layer.Dropout(0.5))
            new_conv_fc.append(layer.Linear(192, p_num_cls, bias=bias_flag))
            break

        elif isinstance(l, nn.ReLU):
            new_conv_fc.append(nn.ReLU())
        elif isinstance(l, neuron.LIFNode):
            new_conv_fc.append(
                neuron.LIFNode(tau=args.tau, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True))
        else:
            raise NotImplementedError(f'Unknown Layer name: {l}')
        conv_idx += 1
            
    logger.info('pruned model architecture: \n' + str(new_conv_fc) + '\nFinish build pruned model architecture')

    old_weights = [net.conv_fc[conv_idx].weight.data for conv_idx in conv_idx_rec]

    for idx, conv_idx in enumerate(conv_idx_rec):

        current_old_weights = old_weights[idx] # C_out, C_in, H, W
        preserve_index = torch.tensor(layer_index[idx])
        new_weights = current_old_weights[preserve_index == 1]  # new_C_out, C_in, H, W
        new_conv_fc[conv_idx].weight = nn.Parameter(new_weights) # assign to pre-conv

        if idx < len(conv_idx_rec) - 1:
            next_weights = torch.permute(old_weights[idx + 1], dims=[1,0,2,3]) # C_out, C_in, H, W -> C_in, C_out, H, W
            next_weights = next_weights[preserve_index == 1] # new_C_in, C_out, H, W
            next_weights = torch.permute(next_weights, dims=[1, 0, 2, 3]) #  new_C_in, C_out, H, W -> C_out, new_C_in, H, W
            old_weights[idx + 1] = next_weights
        
    logger.info('Finish loading pretrained weight to pruned model')

    net.conv_fc = new_conv_fc
    net.to(args.device)

    logger.info('\n' + str(net))
    for l in net.conv_fc:
        if isinstance(l, (layer.Conv2d, layer.Linear)):
            print(f'{l.weight.data.shape}')

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print('Start retraining...')
    max_test_acc = -1
    for epoch in range(args.epochs):
        if epoch + 1 > 20:
            break
        print(f'start epoch {epoch + 1}')
        net.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            out_fr = 0.
            loss = 0.

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):
                    output = net(img[t])
                    out_fr += output
                    loss += F.cross_entropy(output, label)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    output = net(encoded_img)
                    out_fr += output
                    loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        output = net(img[t])
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        output = net(encoded_img)
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net, 
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(
                checkpoint, 
                os.path.join(args.split_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint', f'{pmodel_name}.pth'))
            
        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')

    logger.info('Finish retrain the pruned model')


if args.fusion:
    logger.info('Start fusion model building...')

    splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint/')
    model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
    pmodels = []
    for model_name in model_names:
        logger.info(f'Load {model_name} ')
        pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
        pmodels.append(pth['net'])

    in_dim = np.sum([pmodel.conv_fc[-5].in_features for pmodel in pmodels]) # the first linear layer in feature
    pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels] # each pruned model end before the first linear layer appear
    pmodel_num = len(model_names)
        
    fusion_model = Fusion(in_dim, pmodel_num, args.num_cls, args.act)
    fusion_model.to(args.device)

    for pmodel in pmodels:
        for param in pmodel.parameters():
            param.requires_grad = False

    pmodels = [pmodel.to(args.device) for pmodel in pmodels]

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(fusion_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    max_test_acc = -1
    for epoch in range(args.epochs):
        if epoch + 1 > 15:
            break
        print(f'start epoch {epoch + 1}')
        fusion_model.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0
        start_time = time.time()
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            out_fr = 0.
            loss = 0.

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                # for neuro dataset, dataloader generate (N, T, C, H, W), so we need change (T, N, C, H, W)
                img = img.transpose(0, 1)
                for t in range(args.T):
                    fes = torch.concat([pmodel(img[t]) for pmodel in pmodels], dim=1)
                    output = fusion_model(fes)
                    out_fr += output
                    loss += F.cross_entropy(output, label)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    fes = torch.concat([pmodel(encoded_img) for pmodel in pmodels], dim=1)
                    output = fusion_model(fes)
                    out_fr += output
                    loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(fusion_model)
            for pmodel in pmodels:
                functional.reset_net(pmodel)

        train_time = time.time() - start_time
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        fusion_model.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        fes = torch.concat([pmodel(img[t]) for pmodel in pmodels], dim=1)
                        output = fusion_model(fes)  
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        fes = torch.concat([pmodel(encoded_img) for pmodel in pmodels], dim=1)
                        output = fusion_model(fes)  
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(fusion_model)
                for pmodel in pmodels:
                    functional.reset_net(pmodel)
        
        test_loss /= test_samples
        test_acc /= test_samples
        test_time = time.time() - start_time

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': fusion_model,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(fusion_model, f'{splitted_model_dir}/fusion.pt')

        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        logger.info(f'train time: {train_time:.3f}s, test time: {test_time:.3f}s')

    logger.info(f'Max Accuracy for Fusion Model: {max_test_acc:.4f}')
    

if args.infer:
    # infer must be tested on cpu since it will further set to raspberry

    if not os.path.exists('./infer_data/'):
        raise NotImplementedError('extract infer data by yourself first...')

    logger.info('Load data for inference')
    if args.dataset == 'mnist':
        img = torch.load('./infer_data/mnist_frame.pt', map_location='cpu')
    elif args.dataset == 'cifar10':
        img = torch.load('./infer_data/cifar10_frame.pt', map_location='cpu')
    elif args.dataset == 'caltech':
        img = torch.load('./infer_data/caltech_frame.pt', map_location='cpu')
    elif args.dataset == 'cifar10_dvs':
        img = torch.load('./infer_data/cifar10_dvs_frame.pt', map_location='cpu')
    elif args.dataset == 'nmnist':
        img = torch.load('./infer_data/nmnist_frame.pt', map_location='cpu')
    elif args.dataset == 'ncaltech':
        img = torch.load('./infer_data/ncaltech_frame.pt', map_location='cpu')
    elif args.dataset == 'gtzan':
        img = torch.load('./infer_data/gtzan_frame.pt', map_location='cpu')
    else:
        raise NotImplementedError(f'Invalid dataset name: {args.dataset}...')
    
    if args.split:
        # split infer mode
        splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint/')
        pmodels = []
        model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
        fusion_name = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pt')][0]

        for model_name in model_names:
            logger.info(f'Load {model_name} ')
            pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
            pmodels.append(pth['net'])


        pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels]
        fusion_model = torch.load(os.path.join(splitted_model_dir, fusion_name), map_location='cpu')
        
        pmodels = [pmodel.to(args.device) for pmodel in pmodels]
        fusion_model.to(args.device)
        img = img.unsqueeze(0).to(args.device)

        pmodels = [pmodel.eval() for pmodel in pmodels]
        fusion_model.eval()

        with torch.no_grad():
            out_fr = 0.
            infer_time_record = []
            pmodels_infer_time_record = []  # (T, # of pmodel)

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):

                    # 这个用来存t这一时刻的各个模型推理一个frame时间
                    fe_time_rec = []
                    fe_rec = []
                    for pmodel in pmodels:
                        start_time = time.time()
                        out = pmodel(img[t])
                        fe_time_rec.append(time.time() - start_time)
                        fe_rec.append(out)

                    fe = torch.concat(fe_rec, dim=1)

                    elapse_time = max(fe_time_rec)

                    # 这个东西用来存T个时刻各个模型推理一个frame的时间
                    pmodels_infer_time_record.append(fe_time_rec)

                    start_time = time.time()
                    out = fusion_model(fe)
                    elapse_time += (time.time() - start_time)
                    out_fr += out
                    infer_time_record.append(elapse_time)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img

                    fe_time_rec = []
                    fe_rec = []

                    for pmodel in pmodels:
                        start_time = time.time()
                        out = pmodel(encoded_img)
                        fe_time_rec.append(time.time() - start_time)
                        fe_rec.append(out)

                    fe = torch.concat(fe_rec, dim=1)
                    elapse_time = max(fe_time_rec)
                    pmodels_infer_time_record.append(fe_time_rec)

                    start_time = time.time()
                    out = fusion_model(fe)
                    elapse_time += (time.time() - start_time)
                    out_fr += out
                    infer_time_record.append(elapse_time)

            logger.info(f'infer time: {np.mean(infer_time_record):.4f}s')
            pmodels_infer_time_record = np.array(pmodels_infer_time_record)
            # 这个相当于把原本T*device_num的记录平均成1*device_num，也就是代表各个pmodel推一帧样本（平均）要多久
            res = pmodels_infer_time_record.mean(axis=0)
            cankao = [f'{k}: {v:.4f}' for k, v in zip(model_names, res)]
            logger.info(f'average of infer one frame of a sample per pruned model: {np.mean(res):.4f}, max: {np.max(res):.4f}, min:{np.min(res):.4f}, all: {cankao}') 


    else:
        # single mode
        if os.path.exists(os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth')):
            checkpoint = torch.load(os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth'), map_location='cpu')
            net = checkpoint['net']
            logger.info(f'Load existing model')

        net.to(args.device)
        img = img.unsqueeze(0).to(args.device) 

        net.eval() #
        with torch.no_grad():
            out_fr = 0.
            infer_time_record = []
            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):
                    start_time = time.time()
                    out_fr += net(img[t])
                    infer_time_record.append(time.time() - start_time)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    start_time = time.time()
                    out_fr += net(encoded_img)
                    infer_time_record.append(time.time() - start_time)

            print(f'infer time: {np.mean(infer_time_record):.4f}s')


if args.energy:
    del train_dataset, train_data_loader

    if args.split:
        pmodels = []
        splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint/')
        model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
        fusion_name = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pt')][0]

        for model_name in model_names:
            logger.info(f'Load {model_name} ')
            pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
            pmodels.append(pth['net'])

        pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels]
        fusion_model = torch.load(os.path.join(splitted_model_dir, fusion_name), map_location='cpu')

        pmodels = [pmodel.to(args.device) for pmodel in pmodels]
        fusion_model.to(args.device)

        device_num = len(pmodels)

        with torch.no_grad():
            rec = []
            spike_per_model_rec = []

            for img, _ in test_data_loader:
                img = img.to(args.device)

                spike_num = 0.
                b = img.shape[0]
                spike_per_model = [0 for _ in range(device_num)] 

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        fe_rec = []
                        # for pmodel in pmodels:
                        for d in range(device_num):
                            pmodel = pmodels[d]

                            cnt, out = spike_count(pmodel, img[t])

                            spike_per_model[d] += cnt

                            spike_num += cnt
                            fe_rec.append(out)

                        fe = torch.concat(fe_rec, dim=1)
                        cnt, _ = spike_count(fusion_model.mlp, fe)
                        spike_num += cnt
                            
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        fe_rec = []
                        # for pmodel in pmodels:
                        for d in range(device_num):
                            pmodel = pmodels[d]

                            cnt, out = spike_count(pmodel, encoded_img)

                            # 现在存的是各个pmodel推测一个batch内全样本T次后的总脉冲
                            spike_per_model[d] += cnt

                            spike_num += cnt
                            fe_rec.append(out)

                        fe = torch.concat(fe_rec, dim=1)
                        cnt, _ = spike_count(fusion_model.mlp, fe)
                        # 理论上全联接层不应该有spike，暂时设置确保
                        assert cnt == 0, 'you know'
                        spike_num += cnt

                functional.reset_net(fusion_model)

                for pmodel in pmodels:
                    functional.reset_net(pmodel)

                res = spike_num / b
                rec.append(res.cpu())

                # 现在他存的是一个bath内平均单样本T次后的总脉冲
                spike_per_model = [num / b for num in spike_per_model]
                spike_per_model_rec.append(spike_per_model)

        spike_per_model_rec = torch.tensor(spike_per_model_rec).cpu().numpy()  # batch_num * device_num
        # min max mean for pmodels, so need to convert batch_num * device_num to 1 * device_num
        spike_per_model_rec = spike_per_model_rec.mean(axis=0)

        cankao = {f'{k}: {v:2f}' for k, v in zip(model_names, spike_per_model_rec.tolist())}

        logger.info(f'average spikes for one test sample with {args.T} frames in {args.dataset} with architecture {args.act}-{args.arch}: {np.sum(spike_per_model_rec) / device_num:.2f}, min: {np.min(spike_per_model_rec):.2f}, max: {np.max(spike_per_model_rec):.2f}, details per model: {cankao}')

    else:
        if os.path.exists(os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth')):
            checkpoint = torch.load(os.path.join(args.model_dir, f'{args.dataset}_{args.arch}_{args.act}_T{args.T}_checkpoint_max.pth'), map_location='cpu')
            net = checkpoint['net']
            print(f'Load existing model')

        net.to(args.device)
        net.eval()

        with torch.no_grad():
            rec = []
            for img, _ in test_data_loader:
                spike_num = 0.
                img = img.to(args.device)
                b = img.shape[0]

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        cnt, _ = spike_count(net.conv_fc, img[t])
                        spike_num += cnt
                            
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        cnt, _ = spike_count(net.conv_fc, encoded_img)
                        spike_num += cnt

                functional.reset_net(net)

                res = spike_num / b
                rec.append(res.cpu())

        logger.info(f'average spikes for one test samples with {args.T} frames in {args.dataset} with architecture {args.act}-{args.arch}: {np.mean(rec):.2f}')