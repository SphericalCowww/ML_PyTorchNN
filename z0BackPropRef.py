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
def main():
    x = torch.tensor(1.0)
    y = torch.tensor(3.0)
    w = torch.tensor(1.0, requires_grad=True)

    y_hat = w*x
    loss = pow(y_hat - y, 2)

    loss.backward()
    print('')
    print(w)
    print(loss)
    print(w.grad)                       #partial(loss)/partial(w)
    print(loss.grad)
    print(x.grad, y.grad, y_hat.grad)   #all None, because requires_grad=False by default

if __name__ == '__main__': main()






