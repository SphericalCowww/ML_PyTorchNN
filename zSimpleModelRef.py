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
def modelObj(input_dim, output_dim):
    return torch.nn.Linear(input_dim, output_dim)
def main():
    x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
    x_test = torch.tensor([[5]], dtype=torch.float32)
    x_sampleN, x_featureN = x.shape
    y_sampleN, y_featureN = y.shape
 
    learningRate = 0.01
    epochN = 101
    model     = modelObj(x_featureN, y_featureN)  #one layer NN
    lossFunc  = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)   #SGD: stochastic gradient descent
    ###########################################################################################################
    print(f'x_sampleN = {x_sampleN}, x_featureN = {x_featureN}')
    print(f'y_sampleN = {y_sampleN}, y_featureN = {y_featureN}')
    print(f'predict(x_test) = {model(x_test)}')    
    for epochIter in range(epochN):
        y_hat   = model(x)
        lossVal = lossFunc(y, y_hat)
        lossVal.backward()
        optimizer.step()
        optimizer.zero_grad()               #required so that w.grad() is reevaluated every epoch
        weight, bias = model.parameters()
        if epochIter % 10 == 0:
            print(f'epoch {epochIter}: w = {weight[0][0].item():.3f}, loss = {lossVal:.8f}')
            print('  y_hat =', y_hat.detach().numpy().flatten())
    print(f'predict(x_test) = {model(x_test)}') 
    print('done')

if __name__ == '__main__': main()






