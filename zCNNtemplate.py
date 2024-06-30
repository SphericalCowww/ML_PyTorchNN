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
GPUNAME = None
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'
###############################################################################################################
class modelObj(torch.nn.Module):
    def __init__(self, inputSize, classN):
        super().__init__()
        self.layerSquence = torch.nn.Sequential(\
    #############################################################
            torch.nn.Conv2d(3, 10, 5),\
            torch.nn.ReLU(),\
            torch.nn.MaxPool2d(2, stride=2),\
            torch.nn.Conv2d(10, 16, 3),\
            torch.nn.ReLU(),\
            torch.nn.MaxPool2d(2, stride=2),\
            torch.nn.Flatten(),\
            torch.nn.Linear(16*6*6, 100),\
            torch.nn.ReLU(),\
            torch.nn.Linear(100, 40),\
            torch.nn.ReLU(),\
            torch.nn.Linear(40, classN))
            #torch.nn.ReLU())
    # NOTE: input size of the Linear layer need to be specifically evaluated
    # NOTE: the last ReLU activation is NOT needed because it's already included in torch.nn.CrossEntropyLoss()
    #############################################################
    def forward(self, x): return self.layerSquence(x)

def main():
    deviceName = GPUNAME
    epochN     = 20
    batchSize  = 25
    learnRate  = 0.001
    randomSeed = 11 

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizerObj = lambda inputPars: torch.optim.Adam(inputPars, lr=learnRate)

    verbosity   = 2
    printBatchN = 100
    checkpointLoadPath    = 'zCNNtemplate/model1.pth'
    checkpointSavePath    = 'zCNNtemplate/model1.pth'
    tensorboardWriterPath = 'zCNNtemplate/model1'
    pathlib.Path('zCNNtemplate').mkdir(parents=True, exist_ok=True)
    #############################################################
    ### loading data
    torch.manual_seed(randomSeed)
    # NOTE: transform to normalize PILImage range of [0, 1] to [-1, 1]
    transformObj = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
                                                   torchvision.transforms.Normalize([0.5, 0.5, 0.5],\
                                                                                    [0.5, 0.5, 0.5])])
    trainData = torchvision.datasets.CIFAR10(root='./dataset', train=True,  transform=transformObj)
    testData  = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transformObj)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  batch_size=batchSize, shuffle=False)
    dataShape   = trainLoader.dataset.data.shape
    inputSize = dataShape[1]*dataShape[2]
    classN    = len(np.unique(trainLoader.dataset.targets))
    if verbosity >= 1: 
        print('dataShape:', dataShape)
        print('classN   :', classN)
    ### for calculating required
    ''' 
    # check CNN padding and strides
    # Conv2d(input dim (color dim), output size, kernel size)
    # MaxPool2d(kernel size)
    for batchIdx, dataIter in enumerate(trainLoader):
        prop = dataIter[0]
        print(prop.shape)
        prop = torch.nn.Conv2d(3, 10, 5)(prop)
        print(prop.shape)
        prop = torch.nn.MaxPool2d(2, stride=2)(prop)
        print(prop.shape)
        prop = torch.nn.Conv2d(10, 16, 3)(prop)
        print(prop.shape)
        prop = torch.nn.MaxPool2d(2, stride=2)(prop)
        print(prop.shape)
        break
    sys.exit(0)
    '''
    ### training
    if verbosity >= 1: print('using device:', deviceName)
    device    = torch.device(deviceName)       
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
    model.to(device)
    optimizer_to(optimizer, device)
    batchTotalN = len(trainLoader)
    for epoch in range(checkpoint['epoch']+1, epochN):
        correctN = 0
        sampleN  = 0
        lossTot  = 0
        for batchIdx, dataIter in enumerate(tqdm(trainLoader)):
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
            correctN += (predictions == labels).sum().item()  
            if (verbosity >= 1) and ((batchIdx+1)%printBatchN == 0):
                print('epoch:',  str(epoch+1)+'/'+str(epochN)+',',\
                      'step:',   str(batchIdx+1)+'/'+str(batchTotalN)+',',\
                      'loss =', loss.item())
        accuracy = 100.0*(correctN/sampleN)
        checkpoint['epoch']           = epoch
        checkpoint['model_state']     = model.state_dict()
        checkpoint['optimizer_state'] = optimizer.state_dict()
        if verbosity >= 1: print('  accuracy = ', accuracy)
        if checkpointSavePath is not None:
            torch.save(checkpoint, checkpointSavePath)
            if verbosity >= 1: print('  saving:', checkpointSavePath)
        if tensorboardWriterPath is not None:
            tensorboardWriter = SummaryWriter(tensorboardWriterPath)
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
            correctN += (predictions == labels).sum().item() 
    accuracy = 100.0*(correctN/sampleN)
    if verbosity >= 1: print('  accuracy = ', accuracy, '%\ndone')
###############################################################################################################
#https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
###############################################################################################################
if __name__ == '__main__': main()






