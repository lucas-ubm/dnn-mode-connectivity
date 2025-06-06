"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import curves

__all__ = ['WideResNet28x10']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=True)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideBasicCurve(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, fix_points, stride=1):
        super(WideBasicCurve, self).__init__()
        self.bn1 = curves.BatchNorm2d(in_planes, fix_points=fix_points)
        self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True,
                                   fix_points=fix_points)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                   bias=True, fix_points=fix_points)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = curves.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                                          bias=True, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        out = self.dropout(self.conv1(F.relu(self.bn1(x, coeffs_t)), coeffs_t))
        out = self.conv2(F.relu(self.bn2(out, coeffs_t)), coeffs_t)
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x, coeffs_t)
        out += residual

        return out


class WideResNetBase(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=10, dropout_rate=0.):
        super(WideResNetBase, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = nn.Linear(nstages[3], num_classes)
        
        # Pre-allocate memory for the flattened tensor
        self.register_buffer('flatten_shape', torch.zeros(1, nstages[3], dtype=torch.long))

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is in channels_last format
        x = x.contiguous(memory_format=torch.channels_last)
        
        # Forward through convolutional layers
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        
        # Global average pooling and reshape
        out = F.adaptive_avg_pool2d(out, (1, 1))
        batch_size = out.size(0)
        out = out.view(batch_size, -1)
        
        # Linear layer
        out = self.linear(out)
        
        return out


class WideResNetCurve(nn.Module):
    def __init__(self, num_classes, fix_points, depth=28, widen_factor=10, dropout_rate=0.):
        super(WideResNetCurve, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3curve(3, nstages[0], fix_points=fix_points)
        self.layer1 = self._wide_layer(WideBasicCurve, nstages[1], n, dropout_rate, stride=1,
                                       fix_points=fix_points)
        self.layer2 = self._wide_layer(WideBasicCurve, nstages[2], n, dropout_rate, stride=2,
                                       fix_points=fix_points)
        self.layer3 = self._wide_layer(WideBasicCurve, nstages[3], n, dropout_rate, stride=2,
                                       fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(nstages[3], momentum=0.9, fix_points=fix_points)
        self.linear = curves.Linear(nstages[3], num_classes, fix_points=fix_points)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, fix_points):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, fix_points=fix_points, stride=stride)
            )
            self.in_planes = planes

        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        for block in self.layer1:
            out = block(out, coeffs_t)
        for block in self.layer2:
            out = block(out, coeffs_t)
        for block in self.layer3:
            out = block(out, coeffs_t)
        out = F.relu(self.bn1(out, coeffs_t))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out, coeffs_t)

        return out


class WideResNet28x10:
    base = WideResNetBase
    curve = WideResNetCurve
    kwargs = {'depth': 28, 'widen_factor': 10}
