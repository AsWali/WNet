# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:32:16 2018
@author: Tao Lin

Training and Predicting with the W-Net unsupervised segmentation architecture
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from scipy.special import softmax
import matplotlib.pyplot as plt

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import WNet
from utils.crf import dense_crf


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--model', metavar='C', default="model_20210415210220", type=str, 
                    help='name of the saved model')
parser.add_argument('--image', metavar='C', default=None, type=str, 
                    help='path to the image')
parser.add_argument('--squeeze', metavar='K', default=20, type=int, 
                    help='Depth of squeeze layer')

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def show_gray(image):
    image = image.numpy()
    arr = np.asarray(image)
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.show()
    
def main():
    args = parser.parse_args()
    model = WNet.WNet(args.squeeze)

    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

    image = Image.open(args.image).convert('RGB')
    x = transform(image)[None, :, :, :]

    enc, dec = model(x)
    show_image(x[0])
    plt.imshow(torch.argmax(enc, dim = 1).detach()[0])
    plt.show()
    show_image(dec[0, :, :, :].detach())

if __name__ == '__main__':
    main()


# python .\enc_hidden_test.py --image="data/JPEGImages/2008_006002.jpg"