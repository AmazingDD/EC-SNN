from utils import *
import torch.nn as nn
import torch

'''
TODOLIST:
1. more complicated module
2. more encoding ways
'''

def mac_ac_count(layers,data,init_lsar = 1.0):
    total_cnt = 0
    lsar = init_lsar
    inputs = data
    for layer in layers:
        inputs = layer(inputs)
        if isinstance(layer,nn.Conv2d):
            kn1,kn2 = layer.kernel_size
            cn_prev = layer.in_channels
            cn_cur = layer.out_channels
            hn,wn = data.shape[-2:]
            layer_cnt = kn1 * kn2 * hn * wn * cn_cur * cn_prev
            print("layer name:",layer)
            print("spike count:",layer_cnt * lsar)
            print("lsar:",lsar)
            total_cnt += layer_cnt * lsar
        elif isinstance(layer,nn.Linear):
            batch = inputs.shape[0]
            layer_cnt = batch * layer.in_features * layer.out_features
            total_cnt += layer_cnt * lsar
        elif isinstance(layer, (neuron.LIFNode, nn.ReLU)):
            lsar = inputs.count_nonzero() / inputs.numel()
    return total_cnt,inputs