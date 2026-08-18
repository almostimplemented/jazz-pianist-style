#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules, layers, blocks, etc. used by multiple different encoders"""

import torch
import torch.nn as nn

__all__ = [
    "MaskedAvgPool", "LinearFlatten", "Conv1x1", "GeM", "IBN", "ConvLayer", "LinearLayer", "GRU", "TripletMarginLoss",
    "Identity"
]


class MaskedAvgPool(nn.Module):
    """Average pooling that omits dimensions which have been masked with zeros. When masks=3, equivalent to max pool."""

    def __init__(self):
        super(MaskedAvgPool, self).__init__()
        self.dim = 2

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Computes average for all non-masked columns of a 3D tensor. Expected shape is (batch, features, channels)"""
        mask = x != 0
        pooled = (x * mask).sum(dim=self.dim) / mask.sum(dim=self.dim)
        # We add an extra dimension here for parity with AdaptiveAvgPool/MaxPool in torch
        return pooled.unsqueeze(2)


class LinearFlatten(nn.Module):
    """For a tensor of shape (batch, features, channels), returns a tensor of shape (batch, features * channels)"""

    def __init__(self):
        super(LinearFlatten, self).__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # We add an extra dimension here for parity with AdaptiveAvgPool/MaxPool in torch
        return torch.flatten(x.permute(0, 2, 1), start_dim=1).unsqueeze(2)


class Conv1x1(nn.Module):
    """Dimensionality reduction from (batch, channels, features) -> (batch, features) using 1x1 convolution"""

    def __init__(self, in_channels: int = 4, ):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,  # No padding so number of features stays the same
            bias=True  # Nb. the bias term means that we can get a non-zero output even for zero-masked channels
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # (batch, features, channels) -> (batch, channels, features)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # (batch, features, 1)
        return x.permute(0, 2, 1)


class GeM(nn.Module):
    """Compresses a multidimensional input to a fixed-length vector with generalized mean (GeM) pooling."""

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super(GeM, self).__init__()
        # This sets p to a trainable parameter with an initial parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        # Margin value to prevent division by zero
        self.eps = eps

    def forward(self, x) -> torch.tensor:
        """Pools input with GeM"""
        dims = len(x.size())
        # Three dimensions: use 1D pooling
        if dims == 3:
            pooler = nn.functional.avg_pool1d
            kernel_size = x.size(-1)
        # Four dimensions: use 2D pooling
        elif dims == 4:
            pooler = nn.functional.avg_pool2d
            kernel_size = (x.size(-2), x.size(-1))
        else:
            raise ValueError(f"GeM pooling expected either 3 or 4 dimensions, but got {dims}")
        return pooler(x.clamp(min=self.eps).pow(self.p), kernel_size=kernel_size).pow(1. / self.p)


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, a) -> torch.tensor:
        # (batch_size, 512, height, width)
        batch, features, height, width = a.size()
        # (batch_size, width, 512, height)
        a = a.permute(0, 3, 1, 2).contiguous().view(batch, width, features * height)
        # (batch_size, sequence_length, features)
        a, _ = self.gru(a)
        # (batch_size, features, sequence_length)
        a = a.permute(0, 2, 1).contiguous()
        return a


class IBN(nn.Module):
    """Single IBN layer. Batch normalize half the input channels, instance normalize the other."""

    def __init__(self, out_channels: int, ratio: float = 0.5):
        super(IBN, self).__init__()
        self.half = int(out_channels * ratio)
        self.instance_norm = nn.InstanceNorm2d(self.half, affine=True)
        self.batch_norm = nn.BatchNorm2d(out_channels - self.half)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Split tensor along the channel dimension
        split = torch.split(x, self.half, 1)
        #  Batch normalization is applied to half the channels, instance normalization to the other half
        out1 = self.instance_norm(split[0].contiguous())
        out2 = self.batch_norm(split[1].contiguous())
        # Concatenate the results of both normalization procedures
        out = torch.cat((out1, out2), 1)
        return out


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            has_pool: bool = False,
            conv_kernel_size: int = 3,
            pool_kernel_size: int = 2,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(conv_kernel_size, conv_kernel_size),
            stride=1,
            padding=1
        )
        self.drop = nn.Dropout2d(p=0.2)
        self.relu = nn.ReLU()
        self.norm = norm_layer(out_channels)
        self.has_pool = has_pool
        if has_pool:
            self.avgpool = nn.AvgPool2d(kernel_size=(pool_kernel_size, pool_kernel_size), padding=0)

    def __len__(self) -> int:
        """Returns the number of modules in the convolutional layer."""
        return 5 if self.has_pool else 4

    def forward(self, a) -> torch.tensor:
        a = self.conv(a)
        a = self.drop(a)
        a = self.relu(a)
        a = self.norm(a)
        if self.has_pool:
            a = self.avgpool(a)
        return a


class LinearLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.drop = nn.Dropout(p)

    def forward(self, a) -> torch.tensor:
        a = self.fc(a)
        a = self.drop(a)
        return a


class TripletMarginLoss(nn.Module):
    """Simple implementation of triplet loss with a margin for negative mining and optional L2-normalization"""

    def __init__(self, margin: float = 0.3, similarity: str = "cosine"):
        super().__init__()
        self.margin = margin
        assert similarity in ["dot", "cosine"], "similarity must be either 'dot' or 'cosine'"
        # Cosine similarity == dot product of L2-normalized vectors
        self.normalize = True if similarity == "cosine" else False

    def compute_similarity_matrix(self, anchors: torch.tensor, positives: torch.tensor) -> torch.tensor:
        # If we're applying L2 normalization
        if self.normalize:
            anchors = torch.nn.functional.normalize(anchors, p=2, dim=-1)
            positives = torch.nn.functional.normalize(positives, p=2, dim=-1)
        # Compute dot product
        return torch.matmul(anchors, positives.permute(1, 0))

    def compute_loss(self, similarity_matrix: torch.tensor, class_idxs: torch.tensor = None):
        # Triplet loss: normalized, cross entropy is not normalized
        positive_sims = similarity_matrix.diagonal().unsqueeze(1)
        negative_sims = similarity_matrix.clone()
        # Calculate the loss: relu has function of the max(0, sim) operation
        loss = torch.nn.functional.relu(negative_sims - positive_sims + self.margin)
        # Set the diagonals and OTHER CLIPS BY THE SAME ARTIST to 0.
        if class_idxs is not None:
            class_idxs = class_idxs.unsqueeze(0).repeat(class_idxs.size(0), 1)
            mask = (class_idxs != class_idxs.t()).bool()
        # Just set the diagonals (positive similarity) to 0.
        else:
            mask = ~torch.eye(*loss.size(), device=similarity_matrix.device).bool()
        return loss * mask

    def forward(self, anchors: torch.tensor, positives: torch.tensor, class_idxs: torch.tensor = None) -> torch.tensor:
        # Compute (potentially normalized) similarity matrix
        similarity_matrix = self.compute_similarity_matrix(anchors, positives)
        # Compute the loss from the similarity matrix and return the mean
        loss = self.compute_loss(similarity_matrix, class_idxs)
        return loss.mean()


class Identity(torch.nn.Module):
    """Returns the input tensor without modifying. Used to replace classification head on `torchvision` models"""

    def __init__(self, expand_dim: bool = False):
        super(Identity, self).__init__()
        self.expand_dim = expand_dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        # If required, this will add an empty extra dimension
        # For use in DisentangleNet where we want tensors of (batch, 1, features) to concatenate along dim=1
        if self.expand_dim:
            x = x.unsqueeze(1)
        return x
