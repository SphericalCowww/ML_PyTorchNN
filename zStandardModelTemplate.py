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
from torch.utils.tensorboard import SummaryWriter   #tensorboard --logdir ...
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
    device = 'cpu'#torch.device(('cuda' if torch.cuda.is_available else 'cpu'))
    epochN     = 21
    batchSize  = 100
    learnRate  = 0.001
    randomSeed = 11 

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizerObj = lambda inputPars: torch.optim.Adam(inputPars, lr=learnRate)

    verbosity   = 2
    printBatchN = 100
    checkpointLoadPath = 'zStandardModelTemplate/model1.pth'
    checkpointSavePath = 'zStandardModelTemplate/model1.pth'
    tensorboardWriter = SummaryWriter('zStandardModelTemplate/model1')
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
    ### training
    model     = modelObj(inputSize, classN)
    optimizer = optimizerObj(model.parameters())    #Module.parameters() are all parameters to be optimized
    checkpoint = {'epoch': -1}
    if checkpointLoadPath is not None:
        checkpoint = torch.load(checkpointLoadPath)
        model    .load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if verbosity >= 1: 
            print('  loading:', checkpointLoadPath)
            print('    epoch:', checkpoint['epoch']+1)
        if verbosity >= 2: 
            optParDict = optimizer.state_dict()['param_groups'][0]
            for key in optParDict: print('   ', key, ':', optParDict[key])
    else: 
        model     = modelObj(inputSize, classN)
        optimizer = optimizerObj(model.parameters())    #Module.parameters() are all parameters to be optimized
    if tensorboardWriter is not None:
        tensorboardWriter.add_graph(model)
        tensorboardWriter.close()
    batchTotalN = len(trainLoader)
    for epoch in range(checkpoint['epoch']+1, epochN):
        correctN = 0
        sampleN  = 0
        lossTot  = 0
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
            #trend tracking
            lossTot += loss.item()
            predictions = torch.max(outputs, 1)[1]
            sampleN  += labels.shape[0]
            correctN += np.sum(np.array((predictions == labels))) 
            if (verbosity >= 1) and ((batchIdx+1)%printBatchN == 0):
                print('epoch:',  str(epoch+1)+'/'+str(epochN)+',',\
                      'step:',   str(batchIdx+1)+'/'+str(batchTotalN)+',',\
                      'loss =', loss.item())
        accuracy = 100.0*(correctN/sampleN)
        checkpoint['epoch']           = epoch
        checkpoint['model_state']     = model.state_dict()
        checkpoint['optimizer_state'] = optimizer.state_dict()
        if checkpointSavePath is not None:
            torch.save(checkpoint, checkpointSavePath)
            if verbosity >= 1: print('  saving:', checkpointSavePath)
        if tensorboardWriter is not None:
            tensorboardWriter.add_scalar('loss',     lossTot,  epoch)
            tensorboardWriter.add_scalar('accuracy', accuracy, epoch) 
            tensorboardWriter.close()
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
            correctN += np.sum(np.array((predictions == labels))) 
    accuracy = 100.0*(correctN/sampleN)
    if verbosity >= 1: print('accuracy = ', accuracy, '%\ndone')
###############################################################################################################
if __name__ == '__main__': main()






