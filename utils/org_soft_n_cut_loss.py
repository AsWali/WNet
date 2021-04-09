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



## https://github.com/fkodom/wnet-unsupervised-image-segmentation/blob/dd47622695d8cbb4f7d688d0ecce0452eadded7d/src/loss.py
def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel

class NCutLoss2D(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius: int = 5, sigma_1: float = 4, sigma_2: float = 10):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(NCutLoss2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.
        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv2d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv2d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss 