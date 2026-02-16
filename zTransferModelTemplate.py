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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter   #tensorboard --logdir ...

GPUNAME = 'cpu'
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'
###############################################################################################################
from torchvision.models import resnet18, ResNet18_Weights
def modelObj(inputSize, classN):    # note: this is not a class
    weights = ResNet18_Weights.DEFAULT 
    model = resnet18(weights=weights)    

    features = model.fc.in_features
    model.fc = torch.nn.Linear(features, classN)    #automatically set this layer's par.requires_grad = True
    return model
RESNET_MEAN = np.array([0.485, 0.456, 0.406])
RESNET_STD  = np.array([0.229, 0.224, 0.225])
def main():
    deviceName = GPUNAME
    epochN     = 3
    batchSize  = 100
    learnRate  = 0.0001
    randomSeed = 11 

    lossFunction = torch.nn.CrossEntropyLoss()
    #incorporate regularization: stackoverflow.com/questions/42704283
    optimizerObj = lambda inputPars: torch.optim.Adam(inputPars, lr=learnRate, weight_decay=1E-5)
    #schedulerObj = lambda inputOpt, lastEpoch:\ 
    #    torch.optim.lr_scheduler.StepLR(inputOpt, last_epoch=lastEpoch, step_size=2, gamma=0.5)
    schedulerObj = lambda inputOpt, lastEpoch:\
        torch.optim.lr_scheduler.LinearLR(inputOpt, last_epoch=lastEpoch, start_factor=1.0, end_factor=0.5,\
                                          total_iters=20)

    verbosity   = 2
    printBatchN = 100
    pathlib.Path('yTransferModelTemplate').mkdir(parents=True, exist_ok=True)
    checkpointLoadPath    = 'yTransferModelTemplate/model1.pth'
    checkpointSavePath    = 'yTransferModelTemplate/model1.pth'
    tensorboardWriterPath = 'yTransferModelTemplate/model1'
    plotTestBatchN          = 10000
    plotTestSampleNperBatch = 1
    #############################################################
    ### loading data
    torch.manual_seed(randomSeed)
    dataDir   = './dataset/catDog/'
    imageSize = 224
    dataTransformers = {'train': torchvision.transforms.Compose([\
                                    torchvision.transforms.RandomResizedCrop(imageSize),\
                                    torchvision.transforms.RandomHorizontalFlip(),\
                                    torchvision.transforms.ToTensor(),\
                                    torchvision.transforms.Normalize(RESNET_MEAN, RESNET_STD)]),
                        'test':  torchvision.transforms.Compose([\
                                    torchvision.transforms.Resize(imageSize),
                                    torchvision.transforms.CenterCrop(imageSize),
                                    torchvision.transforms.ToTensor(),\
                                    torchvision.transforms.Normalize(RESNET_MEAN, RESNET_STD)]),}    

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
        print('train mapping:', trainData.class_to_idx)
        print('test mapping :', testData.class_to_idx)
    ### training
    if verbosity >= 1: 
        print('using device:', deviceName)
    device    = torch.device(deviceName)       
    model     = modelObj(inputSize, classN)
    optimizer = optimizerObj(model.parameters())    #Module.parameters() are all parameters to be optimized
    checkpoint = {'epoch': -1}
    if checkpointLoadPath is not None:
        if os.path.exists(checkpointLoadPath) == False: 
            checkpointLoadPath = None
            if verbosity >= 1: 
                print('  loading:', checkpointLoadPath)
                print('    epoch:', 0)
        else:
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
        model.train()
        correctN, sampleN, lossTot = 0, 0, 0
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
        ### testing; independent from training
        model.eval()
        correctN, sampleN = 0, 0
        with torch.no_grad():
            for batchIdx, dataIter in enumerate(testLoader):
                samples     = dataIter[0].to(device)
                labels      = dataIter[1].to(device)
                outputs     = model(samples)
                predictions = torch.max(outputs, 1)[1] 
                sampleN  += labels.shape[0]
                correctN += (predictions == labels).sum().item() 
        validation = 100.0*(correctN/sampleN)
        if verbosity >= 1: print('  test validation accuracy = ', validation, '%\n')
        if tensorboardWriterPath is not None:
            tensorboardWriter = SummaryWriter(tensorboardWriterPath)
            tensorboardWriter.add_scalar('loss',       lossTot,  epoch)
            tensorboardWriter.add_scalar('accuracy',   accuracy, epoch)
            tensorboardWriter.add_scalar('validation', validation, epoch)
            tensorboardWriter.flush()
            tensorboardWriter.close()
    model.eval()
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
            if batchIdx < plotTestBatchN:
                for sampleIdx in range(len(samples)): 
                    if plotTestSampleNperBatch <= sampleIdx: break
                    figureName = figureDir + '/testPlot_batch' + str(batchIdx) +'_sample'+str(sampleIdx)+'.png'
                    labelName      = classes[labels[sampleIdx]]
                    predictionName = classes[predictions[sampleIdx]]
                    ### NOTE: depends on color dim and normalization
                    img = samples[sampleIdx].cpu().numpy().transpose((1, 2, 0))
                    img = RESNET_STD*img + RESNET_MEAN      # undo normalization
                    img = np.clip(img, 0, 1)                # ensure values stay between 0 and 1
                    plt.imshow(img)
                    ###
                    plt.title('label: '+labelName+', prediction: '+predictionName)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(figureName)
                    if verbosity >= 1: print('  saving:', figureName)
                    plt.clf()
    validation = 100.0*(correctN/sampleN)
    if verbosity >= 1: print('  final test validation accuracy = ', validation, '%\ndone')
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






