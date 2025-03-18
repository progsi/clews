#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/7/18 8:00 PM
# software: PyCharm

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        return

    def forward(self, x):
        return self.linear_layer(x)


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(Conv1d, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        return

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AttentiveStatisticsPooling(torch.nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    output_channels: int
        The number of output channels.
    """

    def __init__(self, channels, output_channels):
        super().__init__()

        self._eps = 1e-12
        self._linear = Linear(channels * 3, channels)
        self._tanh = torch.nn.Tanh()
        self._conv = Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self._final_layer = torch.nn.Linear(channels * 2, output_channels, bias=False)
        return

    @staticmethod
    def _compute_statistics(x: torch.Tensor, m: torch.Tensor, eps: float, dim: int = 2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
        return mean, std

    def forward(self, x: torch.Tensor):
        """Calculates mean and std for a batch (input tensor).

        Args:
          x : torch.Tensor
              Tensor of shape [N, L, C].
        """

        x = x.transpose(1, 2)
        L = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device)
        mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True).float()

        mean, std = self._compute_statistics(x, mask / total, self._eps)
        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([x, mean, std], dim=1)
        attn = self._conv(
            self._tanh(self._linear(attn.transpose(1, 2)).transpose(1, 2))
        )

        attn = attn.masked_fill(mask == 0, float("-inf"))  # Filter out zero-padding
        attn = F.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        pooled_stats = self._final_layer(torch.cat((mean, std), dim=1))
        return pooled_stats

    def forward_with_mask(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ):
        """Calculates mean and std for a batch (input tensor).

        Args:
          x : torch.Tensor
              Tensor of shape [N, C, L].
          lengths:
        """
        L = x.shape[-1]

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.

        # torch.std is unstable for backward computation
        # https://github.com/pytorch/pytorch/issues/4320
        total = mask.sum(dim=2, keepdim=True).float()
        mean, std = self._compute_statistics(x, mask / total, self._eps)

        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([x, mean, std], dim=1)

        # Apply layers
        attn = self.conv(self._tanh(self._linear(attn, lengths)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)
        return pooled_stats

    @staticmethod
    def length_to_mask(
        length: torch.Tensor,
        max_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Creates a binary mask for each sequence.

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.

        Example
        -------
        """
        assert len(length.shape) == 1

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        if dtype is None:
            dtype = length.dtype

        if device is None:
            device = length.device

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask
