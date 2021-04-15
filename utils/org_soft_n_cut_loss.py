import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import torch.nn.functional as F

def get_weights(flatten_image, o_w, o_h, oi=10, ox=4, radius=5):
    A = flatten_image.expand(len(flatten_image), len(flatten_image))
    AT = A.T
    B = A - AT
    W_F = torch.exp(torch.div(-1*(B **2), oi**2))

    x = torch.Tensor(range(o_w))
    y = torch.Tensor(range(o_h))
    
    if(torch.cuda.is_available()):
        x = x.cuda()
        y = y.cuda()

    X_cord = x.repeat_interleave(o_h)
    Y_cord = y.repeat(o_w)

    X_expand = X_cord.expand(len(X_cord), len(X_cord))
    Y_expand = Y_cord.expand(len(Y_cord), len(Y_cord))
    X_expandT = X_expand.T
    Y_expandT = Y_expand.T
    Xij = (X_expand - X_expandT) ** 2
    Yij = (Y_expand - Y_expandT) ** 2

    sq_distance_matrix = torch.hypot(Xij, Yij)
    mask = sq_distance_matrix.le(radius)

    C = torch.exp(torch.div(-1*(sq_distance_matrix **2), ox**2))
    W_X = torch.mul(mask, C)

    weights = torch.mul(W_F, W_X)

    return weights


def numerator(A, w):
    flatten_a = A.flatten()
    prob = torch.outer(flatten_a, flatten_a)
    a = torch.mul(w, prob)
    return torch.sum(a)

def denominator(A, w):
    flatten_a = A.flatten()
    prob = flatten_a.expand(len(flatten_a), len(flatten_a))
    a = torch.mul(w, prob)
    return torch.sum(a)


def soft_n_cut_loss(image, k, prob):
    soft_n_cut_loss = k

    image = torch.mean(image, dim=0)
    flatten_image = torch.flatten(image)
    weights = get_weights(flatten_image, image.shape[0], image.shape[1])

    for i in range(k):
        soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[i,:,],weights)/denominator(prob[i,:,:],weights))

    return soft_n_cut_loss


def batch_soft_n_cut_loss(input, enc, k):
    loss = 0

    for i in range(input.shape[0]):
        loss += soft_n_cut_loss(input[i], k, enc[i])

    return loss / input.shape[0]

