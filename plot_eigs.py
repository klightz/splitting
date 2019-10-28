import numpy as np
import argparse
import os

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns


parser = argparse.ArgumentParser(description='plotting eigens')
parser.add_argument('--checkpoint', type=str, 
                    help='checkpoint path', required=True)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)

eigen_vals = [ v.cpu().numpy() for v in checkpoint['min_eig_vals']]
vals = np.concatenate(eigen_vals, axis=0)

yvals = np.sort(vals)

xvals = np.arange(len(yvals))
x_10 = np.quantile(xvals, 0.1)
x_20 = np.quantile(xvals, 0.2)
x_30 = np.quantile(xvals, 0.3)


plt.plot(xvals, yvals)
plt.plot([x_10, x_10], [np.min(yvals), np.max(yvals)])
plt.plot([x_20, x_20], [np.min(yvals), np.max(yvals)])
plt.plot([x_30, x_30], [np.min(yvals), np.max(yvals)])

plt.savefig('min_eigen_vals.png')
plt.close()


