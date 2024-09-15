import os, sys, pathlib, time, re, glob, math
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
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
def modelObj(inputSize, classN):    # note: this is not a class
    model = torchvision.models.resnet18(pretrained=True)
    #for par in model.parameters(): par.requires_grad = False    #fix the weights of all layers
    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, classN)    #automatically set this layer's par.requires_grad = True
    return model
def main():
    deviceName = GPUNAME
    epochN     = 1
    batchSize  = 100
    learnRate  = 0.001
    randomSeed = 11 

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizerObj = lambda inputPars: torch.optim.Adam(inputPars, lr=learnRate)
    #schedulerObj = lambda inputOpt, lastEpoch:\ 
    #    torch.optim.lr_scheduler.StepLR(inputOpt, last_epoch=lastEpoch, step_size=2, gamma=0.5)
    schedulerObj = lambda inputOpt, lastEpoch:\
        torch.optim.lr_scheduler.LinearLR(inputOpt, last_epoch=lastEpoch, start_factor=1.0, end_factor=0.5,\
                                          total_iters=20)

    verbosity   = 2
    printBatchN = 100
    checkpointLoadPath    = 'zTransferModelTemplate/model1.pth'
    checkpointSavePath    = 'zTransferModelTemplate/model1.pth'
    tensorboardWriterPath = 'zTransferModelTemplate/model1'
    pathlib.Path('zTransferModelTemplate').mkdir(parents=True, exist_ok=True)
    plotTestBatchN          = 10
    plotTestSampleNperBatch = 10
    #############################################################
    ### loading data
    torch.manual_seed(randomSeed)
    dataDir   = './dataset/catDog/'
    imageSize = 224
    dataTransformers = {'train': torchvision.transforms.Compose([\
                                    torchvision.transforms.RandomResizedCrop(imageSize),\
                                    torchvision.transforms.RandomHorizontalFlip(),\
                                    torchvision.transforms.ToTensor(),\
                                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                        'test':  torchvision.transforms.Compose([\
                                    torchvision.transforms.RandomResizedCrop(imageSize),\
                                    torchvision.transforms.ToTensor(),\
                                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),}    

    trainData = torchvision.datasets.ImageFolder(dataDir+'train', transform=dataTransformers['train'])
    testData  = torchvision.datasets.ImageFolder(dataDir+'test',  transform=dataTransformers['test'])
    classes = ['cat', 'dog']
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  batch_size=batchSize, shuffle=False)
    dataShape = [len(trainData), trainData[0][0].shape[0], trainData[0][0].shape[1]]
    inputSize = dataShape[1]*dataShape[2]
    classN    = len(classes)
    if verbosity >= 1:
        print('dataShape:', dataShape)
        print('classN   :', classN)
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
    if schedulerObj is not None: 
        scheduler = schedulerObj(optimizer, -1 if (checkpoint['epoch'] == -1) else checkpoint['epoch'] + 1)
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
        if schedulerObj is not None:
            scheduler.step()
            if verbosity >= 2:
                optParDict = optimizer.state_dict()['param_groups'][0]
                for key in optParDict: print('   ', key, ':', optParDict[key])
        if checkpointSavePath is not None:
            torch.save(checkpoint, checkpointSavePath)
            if verbosity >= 1: print('  saving:', checkpointSavePath)
        if tensorboardWriterPath is not None:
            tensorboardWriter = SummaryWriter(tensorboardWriterPath)
            tensorboardWriter.add_scalar('loss',     lossTot,  epoch)
            tensorboardWriter.add_scalar('accuracy', accuracy, epoch) 
            tensorboardWriter.close()
        ### testing; independent from training
        correctN, sampleN = 0, 0
        with torch.no_grad():
            for batchIdx, dataIter in enumerate(testLoader):
                samples     = dataIter[0].to(device)
                labels      = dataIter[1].to(device)
                outputs     = model(samples)
                predictions = torch.max(outputs, 1)[1] 
                sampleN  += labels.shape[0]
                correctN += (predictions == labels).sum().item() 
        accuracy = 100.0*(correctN/sampleN)
        if verbosity >= 1: print('  test validation accuracy = ', accuracy, '%\ndone')
    correctN, sampleN = 0, 0
    with torch.no_grad():
        figureDir = tensorboardWriterPath + '_testPlots'
        pathlib.Path(figureDir).mkdir(parents=True, exist_ok=True)
        for batchIdx, dataIter in enumerate(testLoader):
            samples     = dataIter[0].to(device)
            labels      = dataIter[1].to(device)
            outputs     = model(samples)
            predictions = torch.max(outputs, 1)[1]
            sampleN  += labels.shape[0]
            correctN += (predictions == labels).sum().item()
            if deviceName != 'cpu':
                warnings.warn('\nmain(): deviceName needs to be \'cpu\' to generate test plots', Warning)
            if (deviceName == 'cpu') and (batchIdx < plotTestBatchN):
                for sampleIdx in range(len(samples)): 
                    if plotTestSampleNperBatch <= sampleIdx: break
                    figureName = figureDir + '/testPlot_batch' + str(batchIdx) +'_sample'+str(sampleIdx)+'.png'
                    labelName      = classes[labels[sampleIdx]]
                    predictionName = classes[predictions[sampleIdx]]
                    plt.imshow(samples[sampleIdx][0], cmap='gray')
                    plt.title('label: '+labelName+', prediction: '+predictionName)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(figureName)
                    if verbosity >= 1: print('  saving:', figureName)
                    plt.clf()
    accuracy = 100.0*(correctN/sampleN)
    if verbosity >= 1: print('  final test validation accuracy = ', accuracy, '%\ndone')
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






