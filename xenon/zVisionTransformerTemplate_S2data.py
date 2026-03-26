import os, sys, pathlib, time, re, glob, math, datetime
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
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
            
            pool='mean')
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
    def __init__(self, file_path1, file_path2, class_names=['class1', 'class2'], ratio_range=[0.0, 1.0], 
                 rand_seed=0, transform=None):
        if (0.0 <= ratio_range[0] < ratio_range[1] <= 1.0) is False:
            raise AssertionError("pickleData_2Darrays(): invalid ratio_range") 
        data1, data2 = None, None
        with open(file_path1, 'rb') as inputFile:
            data1 = pickle.load(inputFile) 
        with open(file_path2, 'rb') as inputFile:
            data2 = pickle.load(inputFile)
        dataComb  = np.concatenate([data1, data2], axis=0)
        labelComb = np.concatenate([[0]*len(data1), [1]*len(data2)])
        
        idxShuffle = np.arange(len(dataComb))
        np.random.seed(rand_seed) 
        np.random.shuffle(idxShuffle)
        dataComb  = dataComb [idxShuffle]
        labelComb = labelComb[idxShuffle]

        idxStart = int(ratio_range[0]*len(dataComb))
        idxEnd   = int(ratio_range[1]*len(dataComb))
        self.data         = dataComb[ int(ratio_range[0]*len(dataComb)):int(ratio_range[1]*len(dataComb))]
        self.labels       = labelComb[int(ratio_range[0]*len(dataComb)):int(ratio_range[1]*len(dataComb))]
        self.class_to_idx = {class_names[0]: 0, class_names[1]: 1}
        self.transform    = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img   = self.data[idx].astype(np.float32)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
class S2DataTransform:
    def __init__(self, PMTstds, ampThres=0.01, is_training=True,
                 amp_argm_level=0.1, gain_argm_level=0.01, width_argm_level=0.01, shift_argm_level=16):
        self.PMTstds     = PMTstds
        self.ampThres    = ampThres
        self.is_training = is_training
        self.amp_argm_level   = amp_argm_level
        self.gain_argm_level  = gain_argm_level
        self.width_argm_level = width_argm_level
        self.shift_argm_level = shift_argm_level
        if not isinstance(self.PMTstds, torch.Tensor):
            self.PMTstds = torch.from_numpy(self.PMTstds).float().view(1, 1, -1)
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        ### normalization
        if img.ndim == 2:
            img = img.unsqueeze(0) # [1, time, PMT]
        img[img < self.ampThres] = 0
        img = torch.log1p(torch.clamp(img, min=0))
        img = img/(self.PMTstds + 1e-6)
        ### argumentations
        if self.is_training is True:
            ### argumentation: amp
            amp_argm = torch.randn_like(img)*self.amp_argm_level*img
            zero_mask = (img > 0).float()
            img = img + amp_argm*zero_mask
            ### argumentation: gain 
            gain_argm = 1.0 + torch.randn(1).item()*self.gain_argm_level
            img = img*gain_argm
            ### argumentation: width
            width_argm = 1.0 + torch.randn(1).item()*self.width_argm_level
            width_argm = max(0.8, min(1.2, width_argm))
            width_orig = img.shape[1]
            width_new  = int(width_argm*width_orig)
            img = img.unsqueeze(0) 
            img = torch.nn.functional.interpolate(img, size=(width_new, img.shape[-1]), mode='bilinear', align_corners=False)
            img = img.squeeze(0)
            if width_new > width_orig:
                bin_start = (width_new - width_orig) // 2
                img = img[:, bin_start:bin_start+width_orig, :]
            elif width_new < width_orig:
                pad_left  = (width_orig - width_new) // 2
                pad_right = (width_orig - width_new) - pad_left
                img = torch.nn.functional.pad(img, (0, 0, pad_left, pad_right))
            ### argumentation: shift
            shift_argm = np.random.randint(-self.shift_argm_level, self.shift_argm_level)
            img = torch.roll(img, shifts=shift_argm, dims=1)
        ### limit
        img = torch.clamp(img, 0, 10)
        return img
###############################################################################################################
def main():
    verbosity  = 2
    randomSeed = 11

    epochN     = 100
    batchSize  = 64
    learningRate  = 0.0005

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
    dataDir     = os.path.expanduser('~/fuse/dataset/S2data/')
    with open(dataDir+'Kr83m_sim_sample__10000_s2_data_long_per_channel_0pad_PMTstds.pkl', 'rb') as pickleFile:
        imageSTD = pickle.load(pickleFile)
    classes = ['Kr83m', 'fake']
    trainData = pickleData_2Darrays(dataDir+'Kr83m_sim_sample__10000_s2_data_long_per_channel_0pad.pkl', 
                                    dataDir+'fake_event_sim_sample__10001_s2_data_long_per_channel_0pad.pkl', 
                                    class_names=classes, ratio_range=[0.0, 0.9], rand_seed=randomSeed,
                                    transform=S2DataTransform(PMTstds=imageSTD, is_training=True))
    testData  = pickleData_2Darrays(dataDir+'Kr83m_sim_sample__10000_s2_data_long_per_channel_0pad.pkl',  
                                    dataDir+'fake_event_sim_sample__10001_s2_data_long_per_channel_0pad.pkl',  
                                    class_names=classes, ratio_range=[0.9, 1.0], rand_seed=randomSeed,
                                    transform=S2DataTransform(PMTstds=imageSTD, is_training=False))
    loaderArgs = {'batch_size': batchSize}
    if GPUNAME == 'cuda':
        loaderArgs = {'batch_size': batchSize, 'num_workers': 8, 'pin_memory': True}
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, shuffle=True,  **loaderArgs)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  shuffle=False, **loaderArgs)
    if verbosity >= 1:
        print('train_dataShape:', trainData[0][0].shape)
        print('test_dataShape :', testData[0][0].shape)
        print('classN           :', len(classes))
        print('train mapping    :', trainData.class_to_idx)
        print('test mapping     :', testData.class_to_idx)
    #############################################################
    ### monitoring
    printBatchN             = 100
    plotTestBatchN          = 10
    plotTestSampleNperBatch = 10
    pathlib.Path('yVisionTransformerTemplate').mkdir(parents=True, exist_ok=True)
    modelName = datetime.datetime.now().strftime('%y%m%d_%H%M%S')+'model_S2data'
    checkpointLoadPath    = 'yVisionTransformerTemplate_S2data/' + modelName + '.pth'
    checkpointSavePath    = 'yVisionTransformerTemplate_S2data/' + modelName + '.pth'
    tensorboardWriterPath = 'yVisionTransformerTemplate_S2data/' + modelName
    wandbObj = wandb.init(entity='tinglin194-universit-t-m-nster',
                          project='yVisionTransformerTemplate',
                          dir='yVisionTransformerTemplate_S2data/wandbLog',
                          id=modelName,
                          resume='allow',
                          config={'train_dataShape': trainData[0][0].shape,
                                  'test_dataShape':  testData[0][0].shape,
                                  'learningRate': learningRate,
                                  'architecture': 'ViT',
                                  'randomSeed':   randomSeed,
                                  'batchSize':    batchSize,
                                  'dataset':      dataDir},)
    #############################################################
    ### training
    if verbosity >= 1: print('using device:', GPUNAME)
    device = torch.device(GPUNAME)       
    model = modelObj(inputDim=trainData[0][0].shape[-1],
                     channelN=1,
                     patchDim=16,
                     embedDim=384,
                     clsN=len(classes),
                     
                     attnHeadN=4,
                     qkvBias=True,
                     mlpRatio=4.0,
                     attnDropProb=0.3,
                     outputDropProb=0.3,
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






