import os, sys, pathlib, time, re, glob, math, datetime
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
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter   #tensorboard --logdir ...
import wandb

GPUNAME = 'cpu'
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'
###############################################################################################################
from vit_pytorch import ViT
class modelObj(torch.nn.Module):    #cls: output class
    def __init__(self,\
                 inputDim=384,\
                 channelN=3,\
                 patchDim=16,\
                 embedDim=768,\
                 clsN=10,\

                 attnHeadN=12,\
                 qkvBias=True,\
                 mlpRatio=4.0,\
                 attnDropProb=0.0,\
                 outputDropProb=0.0,\
                 blockDepth=12):
        super().__init__()
        if qkvBias == False:
            warnings.warn('modelObj(): vit_pytorch assumes qkvBias == True', Warning)      
        self.vit = ViT(\
            image_size=inputDim,\
            channels=channelN,\
            patch_size=patchDim,\
            dim=embedDim,\
            num_classes=clsN,\
            
            heads=attnHeadN,\
            mlp_dim=int(inputDim*mlpRatio),\
            dropout=attnDropProb,\
            emb_dropout=outputDropProb,\
            depth=blockDepth,\
            
            pool='cls')
    def forward(self, x):
        return self.vit(x)

class MultiHeadViT(torch.nn.Module):
    def __init__(self, clsN=10, feature_heads=['area','amplitude','center_time']):
        super().__init__()
        self.vit = ViT(
            image_size=256,
            channels=3,
            patch_size=16,
            num_classes=0  # don't use ViT's built-in classifier
        )
        embed_dim = self.vit.dim  # usually 768
        
        # create a separate MLP for each head
        self.classifier = torch.nn.Linear(embed_dim, clsN)
        self.feature_heads = torch.nn.ModuleDict({
            name: nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim//2),
                torch.nn.GELU(),
                torch.nn.Linear(embed_dim//2, 1)
            ) for name in feature_heads
        })
        
    def forward(self, x):
        x = self.vit(x)  # returns embedding for [CLS] token
        out_cls = self.classifier(x)
        out_features = {name: head(x) for name, head in self.feature_heads.items()}
        return out_cls, out_features
###############################################################################################################
from torch.utils.data import Dataset
class pickleData_2Darrays(Dataset):
    def __init__(self, file1, file2, class_names=['class1', 'class2'], transform=None):
        with open(file1, 'rb') as inputFile:
            self.data1 = pickle.load(inputFile) 
        with open(file2, 'rb') as inputFile:
            self.data2 = pickle.load(inputFile)
        self.labels = np.concatenate([[0]*len(self.data1), [1]*len(self.data2)])
        self.data   = np.concatenate([self.data1, self.data2], axis=0)
        self.class_to_idx = {class_names[0]: 0, class_names[1]: 1}
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img   = self.data[idx].astype(np.float32)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
class S2DataTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, img):
        img = torch.from_numpy(img).float()
        if img.ndim == 2:
            img = img.unsqueeze(0) # [1, PMT, time]
        img = torch.log1p(img)
        img = (img - self.mean) / (self.std + 1e-6)
        img = img + torch.randn_like(img) * (self.std/10.0)
        return img
###############################################################################################################
def main():
    verbosity  = 2
    randomSeed = 11

    epochN     = 100
    batchSize  = 64
    learningRate  = 0.0001

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizerObj = lambda inputPars: torch.optim.AdamW(inputPars, lr=learningRate, weight_decay=0.05)
    def schedulerObj(inputOpt, lastEpoch):
        warmup_epochs = 10
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\
            inputOpt, T_max=(epochN - warmup_epochs), eta_min=1e-6)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(\
            inputOpt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        return torch.optim.lr_scheduler.SequentialLR(
            inputOpt, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs],\
            last_epoch=lastEpoch)
    #############################################################
    ### loading data
    torch.manual_seed(randomSeed)
    dataDir   = os.path.expanduser('~/fuse/dataset/S2data/')
    imageSize = 512
    imageMean = 0.0
    imageSTD  = 0.5
    classes = ['Kr83m', 'fake']
    trainData = pickleData_2Darrays(dataDir+'Kr83m_sim_sample__10000_s2_data_long_per_channel_0pad.pkl', 
                                    dataDir+'fake_event_sim_sample__10001_s2_data_long_per_channel_0pad.pkl', 
                                    class_names=classes,transform=S2DataTransform(mean=imageMean,std=imageSTD))
    testData  = pickleData_2Darrays(dataDir+'Kr83m_sim_sample__1000_s2_data_long_per_channel_0pad.pkl',  
                                    dataDir+'fake_event_sim_sample__1001_s2_data_long_per_channel_0pad.pkl',  
                                    class_names=classes,transform=S2DataTransform(mean=imageMean,std=imageSTD))
    loaderArgs = {'batch_size': batchSize}
    if GPUNAME == 'cuda':
        loaderArgs = {'batch_size': batchSize, 'num_workers': 8, 'pin_memory': True}
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, shuffle=True,  **loaderArgs)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  shuffle=False, **loaderArgs)
    dataShape = [len(trainData), trainData[0][0].shape[0], trainData[0][0].shape[1]]
    inputSize = dataShape[1]*dataShape[2]
    classN    = len(classes)
    if verbosity >= 1:
        print('dataShape:', dataShape)
        print('classN   :', classN)
        print('train mapping:', trainData.class_to_idx)
        print('test mapping :', testData.class_to_idx)
    #############################################################
    ### monitoring
    printBatchN             = 100
    plotTestBatchN          = 10
    plotTestSampleNperBatch = 10
    pathlib.Path('yVisionTransformerTemplate').mkdir(parents=True, exist_ok=True)
    checkpointLoadPath    = 'yVisionTransformerTemplate_S2data/model1.pth'
    checkpointSavePath    = 'yVisionTransformerTemplate_S2data/model1.pth'
    tensorboardWriterPath = 'yVisionTransformerTemplate_S2data/model1'
    wandbObj = wandb.init(entity='tinglin194-universit-t-m-nster',\
                          project='yVisionTransformerTemplate',\
                          dir='yVisionTransformerTemplate_S2data/wandbLog',\
                          id=datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'model_S2data',\
                          resume='allow',\
                          config={'learningRate': learningRate,\
                                  'architecture': 'ViT',\
                                  'randomSeed':   randomSeed,\
                                  'batchSize':    batchSize,\
                                  'dataset':      dataDir,\
                                  'inputSize':    imageSize,},)
    #############################################################
    ### training
    if verbosity >= 1: print('using device:', GPUNAME)
    device = torch.device(GPUNAME)       
    model = modelObj(inputDim=imageSize,\
                     channelN=1,\
                     patchDim=16,\
                     embedDim=384,\
                     clsN=classN,\
                     
                     attnHeadN=4,\
                     qkvBias=True,\
                     mlpRatio=4.0,\
                     attnDropProb=0.1,\
                     outputDropProb=0.1,\
                     blockDepth=6)
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
    model.to(device)
    optimizer_to(optimizer, device)
    if schedulerObj is not None:
        scheduler = schedulerObj(optimizer, -1 if (checkpoint['epoch'] == -1) else checkpoint['epoch'] + 1)
    batchTotalN = len(trainLoader)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(checkpoint['epoch']+1, epochN):
        model.train()
        correctN, sampleN, lossTot = 0, 0, 0
        for batchIdx, dataIter in enumerate(tqdm(trainLoader)):
            samples = dataIter[0].to(device, non_blocking=True)
            labels  = dataIter[1].to(device, non_blocking=True)
            ### forward
            with torch.amp.autocast('cuda', enabled=(GPUNAME == 'cuda')):
                outputs = model(samples) 
                loss    = lossFunction(outputs, labels)
            ### backward
            optimizer.zero_grad(set_to_none=True)   #required so that w.grad() is reevaluated every epoch
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ### trend tracking
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
                with torch.amp.autocast('cuda', enabled=(GPUNAME == 'cuda')):
                    outputs     = model(samples)
                    predictions = torch.max(outputs, 1)[1] 
                    sampleN  += labels.shape[0]
                    correctN += (predictions == labels).sum().item()
        validation = 100.0*(correctN/sampleN)
        if verbosity >= 1: print('  test validation accuracy = ', validation, '%\n')
        if tensorboardWriterPath is not None:
            tensorboardWriter = SummaryWriter(tensorboardWriterPath)
            tensorboardWriter.add_scalar('loss_tot',   lossTot,                  epoch)
            tensorboardWriter.add_scalar('loss_train', lossTot/len(trainLoader), epoch)
            tensorboardWriter.add_scalar('accuracy',   accuracy,                 epoch)
            tensorboardWriter.add_scalar('validation', validation,               epoch)
            tensorboardWriter.flush()
            tensorboardWriter.close()
            wandbObj.log({'loss_tot':   lossTot,\
                          'loss_train': lossTot/len(trainLoader),\
                          'accuracy':   accuracy,\
                          'validation': validation,})
    wandbObj.finish()
    #############################################################
    ### evaluating
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
            if (batchIdx < plotTestBatchN) or (len(testLoader)-plotTestBatchN < batchIdx):
                for sampleIdx in range(len(samples)): 
                    if plotTestSampleNperBatch <= sampleIdx: break
                    figureName = figureDir + '/testPlot_batch' + str(batchIdx) +'_sample'+str(sampleIdx)+'.png'
                    labelName      = classes[labels[sampleIdx]]
                    predictionName = classes[predictions[sampleIdx]]
                    ### NOTE: depends on color dim and normalization
                    plt.imshow(np.transpose((np.array(samples[sampleIdx].cpu())+1)/2, (1, 2, 0)))
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






