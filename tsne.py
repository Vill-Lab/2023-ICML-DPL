import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import json

from models import *
from utils import progress_bar
import utils

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os

import numpy as np
import pandas as pd
# from tsne import bh_sne
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import openTSNE


parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./tsne_results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=4, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose([   
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = torchvision.datasets.CIFAR10(root="data/", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# set net
net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# # net = GoogLeNet()
# # net = DenseNet121()
# # net = ResNeXt29_2x64d()
# # net = MobileNet()
# # net = MobileNetV2()
# # net = DPN92()
# # net = ShuffleNetG2()
# # net = SENet18()
# # net = ShuffleNetV2(1)
# # net = EfficientNetB0()
# # net = RegNetX_200MF()
# # net = SimpleDLA()

if device == 'cuda':
    net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load("checkpoint/ckpt_vgg_ip0005_x500.pth")
net.load_state_dict(checkpoint['net'])

def gen_features():
    #net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs).cpu()
            outputs = torch.squeeze(outputs)
            outputs_np = outputs.data.numpy()
            
            targets_list.append(targets_np)#[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)#.astype(np.float64)
    print(outputs)

    return targets, outputs


targets, outputs = gen_features()

outputs = np.array(outputs)
targets = np.array(targets)


tsne = openTSNE.TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

embedding = tsne.fit(outputs)


utils.plot(args.save_dir, embedding, targets, colors=utils.MACOSKO_COLORS)

