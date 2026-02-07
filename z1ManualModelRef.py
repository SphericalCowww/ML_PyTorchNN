import os, sys, pathlib, time, re, glob, math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from tqdm import tqdm
import torch

###############################################################################################################
def forwardFunc(w, x):
    return w * x
def lossFunc(y, y_hat):
    return (pow(y - y_hat, 2)).mean()
def main():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    x_test = torch.tensor([[5]], dtype=torch.float32)

    learningRate = 0.01
    epochN = 101
    ###########################################################################################################
    print(f'predict(x_test) = {forwardFunc(w, x_test)}')
    for epochIter in range(epochN):
        y_hat   = forwardFunc(w, x)
        lossVal = lossFunc(y, y_hat)
        lossVal.backward()
        with torch.no_grad():                   #required so that w.grad() doesn't change while updating w
            w -= learningRate * w.grad
        w.grad.zero_()                          #required so that w.grad() is reevaluated every epoch
        if epochIter % 10 == 0:
            print(f'epoch {epochIter}: w = {w:.3f}, loss = {lossVal:.8f}')
            print('  y_hat =', y_hat.detach().numpy(), "\n")
    print(f'predict(x_test) = {forwardFunc(w, x_test)}')
    print('done')

if __name__ == '__main__': main()






