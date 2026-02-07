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
import torchvision
GPUNAME = None
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'
###############################################################################################################
class modelObj(torch.nn.Module):
    def __init__(self, inputSize, classN):
        super().__init__()
        self.layerSquence = torch.nn.Sequential(\
    #############################################################
            torch.nn.Flatten(),
            torch.nn.Linear(inputSize, 100),\
            torch.nn.ReLU(),\
            torch.nn.Linear(100, classN))
    #############################################################
    def forward(self, x): return self.layerSquence(x)

def main():
    device = torch.device('cpu')  #GPUNAME)
    epochN     = 3
    batchSize  = 100
    learnRate  = 0.001
    randomSeed = 11 

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizerObj = lambda inputPars: torch.optim.Adam(inputPars, lr=learnRate)

    verbosity   = 1
    printBatchN = 100
    #############################################################
    ### loading data
    torch.manual_seed(randomSeed) 
    trainData = torchvision.datasets.MNIST(root='./dataset', train=True,\
                                           transform=torchvision.transforms.ToTensor())#, download=True)
    testData  = torchvision.datasets.MNIST(root='./dataset', train=False,\
                                           transform=torchvision.transforms.ToTensor())
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  batch_size=batchSize, shuffle=False)
    dataShape   = trainLoader.dataset.data.shape
    inputSize = dataShape[1]*dataShape[2]
    classN    = len(np.unique(trainLoader.dataset.targets))
    
    '''
    figureName = 'samplePlot.png'
    for batchIdx, dataIter in enumerate(trainLoader):
        samples = dataIter[0]
        labels  = dataIter[1]
        for plotIdx in range(12):
            plt.subplot(3, 4, plotIdx+1)
            plt.imshow(samples[plotIdx][0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(figureName)
        if verbosity >= 1: print('  saving:', figureName)
        break
    '''
    ### training
    model     = modelObj(inputSize, classN)
    optimizer = optimizerObj(model.parameters())    #Module.parameters() are all parameters to be optimized
    batchTotalN = len(trainLoader)
    for epoch in range(epochN):
        for batchIdx, dataIter in enumerate(trainLoader):
            samples = dataIter[0].to(device)
            labels  = dataIter[1].to(device)
            #forward
            outputs = model(samples) 
            loss    = lossFunction(outputs, labels)
            #backward
            optimizer.zero_grad()                   #required so that w.grad() is reevaluated every epoch
            loss.backward()
            optimizer.step()
            if (verbosity >= 1) and ((batchIdx+1)%printBatchN == 0):
                print('epoch:',  str(epoch+1)+'/'+str(epochN)+',',\
                      'step:',   str(batchIdx+1)+'/'+str(batchTotalN)+',',\
                      'loss =', loss.item())
    ### testing
    with torch.no_grad():
        correctN = 0
        sampleN  = 0
        for batchIdx, dataIter in enumerate(testLoader):
            samples     = dataIter[0].to(device)
            labels      = dataIter[1].to(device)
            outputs     = model(samples)
            predictions = torch.max(outputs, 1)[1] 
            sampleN  += labels.shape[0]
            correctN += np.sum(np.array((predictions == labels))) #(predictions == labels).sum().item()
    accuracy = 100.0*(correctN/sampleN)
    if verbosity >= 1: print('accuracy = ', accuracy, '%\ndone')
###############################################################################################################
if __name__ == '__main__': main()






