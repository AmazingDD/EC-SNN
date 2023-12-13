import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer, neuron, surrogate


cfg = {
	'vgg5' : [64, 'P', 128, 128, 'P'],
	'vgg9': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'],
    'vgg19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
    'vgg16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'cifarnet': [256, 256, 256, 'P', 256, 256, 256, 'P']
}

class VGG(nn.Module):
    def __init__(self, act='ann', arch='vgg9', num_cls=10, input_dim=3, img_height=0, img_width=0, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        print(f'------------- {arch.upper()} -------------')

        bias_flag = False if act == 'snn' else True

        h_dim = img_height 
        w_dim = img_width 

        conv = []
        in_channel = input_dim
        out_channel = None
        self.conv_fc = None

        assert arch in cfg.keys(), f'Invalid architecture option {arch}'

        for x in cfg[arch]:
            if x == 'P':
                conv.append(layer.AvgPool2d(kernel_size=2) if act == 'snn' else layer.MaxPool2d(kernel_size=2))
                h_dim //= 2
                w_dim //= 2
            elif x == 'D':
                conv.append(layer.Dropout(0.2))
            else:
                out_channel = x
                conv.append(layer.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=bias_flag))
                conv.append(layer.BatchNorm2d(out_channel))
                conv.append(spiking_neuron(**deepcopy(kwargs)) if act == 'snn' else nn.ReLU())  # ann for relu
                in_channel = out_channel

        if arch in ['vgg5', 'vgg9']:
            self.conv_fc = nn.Sequential(
                *conv,

                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(h_dim * w_dim * in_channel, 1024, bias=bias_flag),
                layer.Dropout(0.5),
                layer.Linear(1024, num_cls, bias=bias_flag),

            )
        elif arch in ['vgg16', 'vgg19']:
            self.conv_fc = nn.Sequential(
                *conv,

                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(h_dim * w_dim * in_channel, 4096, bias=bias_flag),
                layer.Dropout(0.5),
                layer.Linear(4096, 4096, bias=bias_flag),
                layer.Dropout(0.5),
                layer.Linear(4096, num_cls, bias=bias_flag),

            )
        else:
            self.conv_fc = nn.Sequential(
                *conv,
                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(h_dim * w_dim * in_channel, 384, bias=bias_flag),
                layer.Dropout(0.5),
                layer.Linear(384, 192, bias=bias_flag),
                layer.Dropout(0.5),
                layer.Linear(192, num_cls, bias=bias_flag),
            )

    def forward(self, x):
        return self.conv_fc(x)

class Fusion(nn.Module):
    def __init__(self, in_dim, pmodel_num, num_cls, act='snn'):
        super().__init__()
        bias_flag = True if act == 'ann' else False

        self.mlp = nn.Sequential(
            layer.Linear(in_dim, 384 * pmodel_num, bias=bias_flag),
            layer.Dropout(0.5),
            layer.Linear(384 * pmodel_num, 192 * pmodel_num, bias=bias_flag),
            layer.Dropout(0.5),
            layer.Linear(192 * pmodel_num, num_cls * pmodel_num, bias=bias_flag),
            layer.Linear(num_cls * pmodel_num, num_cls, bias=bias_flag),
        )

    def forward(self, x):
        return self.mlp(x)