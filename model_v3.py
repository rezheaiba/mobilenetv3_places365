"""
# @Author  : rezheaiba
# @Update  : 修改了v3的激活层，为配合ipu使用
"""

from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
            # self, in_channel, out_channel, kernel_size=3, stride=1, groups=1
    ) -> None:

        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class SqueezeExcitation(torch.nn.Module):
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
            scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            # hardSigmoid减少计算量，相比于Sigmoid
            # se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
            se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Sigmoid),
    ):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        '''activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU'''
        activation_layer = nn.ReLU6 if cnf.use_hs else nn.ReLU

        # expand 
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )

        if cnf.use_se:
            squeeze_channel = _make_divisible(cnf.expanded_channels // 4, 8)  # min_vlaue=None=division
            layers.append(se_layer(cnf.expanded_channels, squeeze_channel))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None
            )
        )
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor):
        result = self.block(input)
        if self.use_res_connect:  # self.use_res_connect = (cnf.stride == 1 and cnf.input_channels == cnf.out_channels)
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            num_classes_1: int = 1000,
            num_classes_2: int = 1000,
            width_mult: float = 1.0,
            dilated: bool = False,
            reduced_tail: bool = False,
            dropout: float = 0.2,
            arch: str = "mobilenet_v3_large"
    ) -> None:
        super().__init__()

        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)  # 把函数和参数分离并传参
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

        if arch == "mobilenet_v3_large":
            # input_c, kernel, expanded_c, out_c, use_se, activation, stride, dilation
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1,
                           dilation),
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1,
                           dilation),
            ]
            last_channel = adjust_channels(1280 // reduce_divider)  # C5
        elif arch == "mobilenet_v3_small":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1,
                           dilation),
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1,
                           dilation),
            ]
            last_channel = adjust_channels(1024 // reduce_divider)  # C5

        else:
            raise ValueError(f"Unsupported model type {arch}")

        block = InvertedResidual
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,  # 16
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channel = inverted_residual_setting[-1].out_channels
        lastconv_output_channel = 6 * lastconv_input_channel
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channel,
                lastconv_output_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier1 = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            nn.ReLU6(inplace=False),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes_1)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            nn.ReLU6(inplace=False),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes_2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return {
            'classifier1': self.classifier1(x),
            'classifier2': self.classifier2(x),
        }
