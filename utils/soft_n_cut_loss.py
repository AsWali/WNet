import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple


def soft_n_cut_loss_single_k(input, enc, batch_size, k, img_size=(64, 64), ox=4, radius=5 ,oi=10):
    channels = 1
    image = torch.mean(input, dim=1, keepdim=True)
    h, w = img_size
    p = radius

    image = F.pad(input=image, pad=(p, p, p, p), mode='constant', value=99999)
    encoding = F.pad(input=enc, pad=(p, p, p, p), mode='constant', value=99999)

    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1

    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    seg = encoding.unfold(2, kh, dh).unfold(3, kw, dw)

    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    seg = seg.contiguous().view(batch_size, channels, -1, kh, kw)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.view(-1, channels, kh, kw)
    seg = seg.permute(0, 2, 1, 3, 4)
    seg = seg.view(-1, channels, kh, kw)

    center_values = patches[:, :, radius, radius]
    center_values = center_values[:, :, None, None]
    center_values = center_values.expand(-1, -1, kh, kw)

    k_row = (torch.arange(1, kh + 1) - torch.arange(1, kh + 1)[radius]).expand(kh, kw)

    if torch.cuda.is_available():
        k_row = k_row.cuda()

    distance_weights = (k_row ** 2 + k_row.T**2)

    mask = distance_weights.le(radius)
    distance_weights = torch.exp(torch.div(-1*(distance_weights), ox**2))
    distance_weights = torch.mul(mask, distance_weights)
    # the 0's from the padding may cause an issue
    patches = torch.exp(torch.div(-1*((patches - center_values)**2), oi**2))
    W = torch.mul(patches, distance_weights)
    d = W * seg

    nominator = torch.sum(enc * torch.sum(d, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))
    denominator = torch.sum(enc * torch.sum(W, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))

    return torch.div(nominator, denominator)

def soft_n_cut_loss(image, enc, img_size):
    loss = []
    batch_size = image.shape[0]
    k = enc.shape[1]
    for i in range(0, k):
        loss.append(soft_n_cut_loss_single_k(image, enc[:, (i,), :, :], batch_size, k, img_size))
    da = torch.stack(loss)
    return torch.mean(k - torch.sum(da, dim=0))
