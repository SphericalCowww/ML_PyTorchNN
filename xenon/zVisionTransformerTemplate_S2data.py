import os, sys, pathlib, time, re, glob, math, datetime
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pickle, h5py
import random
import matplotlib.pyplot as plt
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
    def __init__(self, file_paths0, file_paths1, class_names=['class1', 'class2'], 
                 ratio_range=[0.0, 1.0], rand_seed=0, transform=None):
        if (0.0 <= ratio_range[0] < ratio_range[1] <= 1.0) is False:
            raise AssertionError("pickleData_2Darrays(): invalid ratio_range") 
        if isinstance(file_paths0, str): file_paths0 = [file_paths0]
        if isinstance(file_paths1, str): file_paths1 = [file_paths1]
        
        randGen = np.random.default_rng(rand_seed)
        data1 = self._load_multiple_files(file_paths0, randGen, ratio_range)
        data2 = self._load_multiple_files(file_paths1, randGen, ratio_range)
        self.data   = np.concatenate([data1, data2], axis=0)
        self.labels = np.concatenate([[0]*len(data1), [1]*len(data2)])
        self.class_to_idx = {class_names[0]: 0, class_names[1]: 1}
        self.transform = transform
        del data1, data2
    def _load_multiple_files(self, paths, randGen, ratio_range):
        accumulated_data = []
        for file_path in paths:
            with open(file_path, 'rb') as inputFile:
                raw_data = pickle.load(inputFile)
                sampleN = len(raw_data)
                shuffleIdx = randGen.permutation(sampleN)
                shuffleIdx = shuffleIdx[int(ratio_range[0]*sampleN):int(ratio_range[1]*sampleN)]
                single_data = np.asarray([raw_data[sampleIdx] for sampleIdx in shuffleIdx], dtype=np.float32)
                del raw_data
                accumulated_data.append(single_data)
                del single_data
        return np.concatenate(accumulated_data, axis=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img   = self.data[idx] 
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
class lazyH5Data_2Darrays(Dataset):
    def __init__(self, file_paths0, file_paths1, class_names=['class1', 'class2'], 
                 ratio_range=[0.0, 1.0], rand_seed=0, transform=None):
        if not (0.0 <= ratio_range[0] < ratio_range[1] <= 1.0):
            raise AssertionError("LazyH5Data_2Darrays(): invalid ratio_range") 
        if isinstance(file_paths0, str): file_paths0 = [file_paths0]
        if isinstance(file_paths1, str): file_paths1 = [file_paths1]
        
        self.dataName     = 'waveforms'
        self.randGen      = np.random.default_rng(rand_seed)
        self.transform    = transform
        self.class_to_idx = {class_names[0]: 0, class_names[1]: 1}
        
        self.index_map = []
        self._build_index(file_paths0, label=0, ratio_range=ratio_range)
        self._build_index(file_paths1, label=1, ratio_range=ratio_range)
        self.file_handles = {}
    def _build_index(self, paths, label, ratio_range):
        for file_path in paths:
            with h5py.File(file_path, 'r') as inputFile:
                sampleN = inputFile[self.dataName].shape[0]
                shuffleIdx = self.randGen.permutation(sampleN)
                shuffleIdx = shuffleIdx[int(ratio_range[0]*sampleN):int(ratio_range[1]*sampleN)]
                for sampleIdx in shuffleIdx:
                    self.index_map.append((file_path, sampleIdx, label))
    def __len__(self):
        return len(self.index_map)
    def __getitem__(self, fileIdx):
        file_path, sampleIdx, label = self.index_map[fileIdx]
        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, 'r')
        file_handle = self.file_handles[file_path]
        img = file_handle[self.dataName][sampleIdx].astype(np.float32)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float()
            if img.ndim == 2:
                img = img.unsqueeze(0) # [1, time, PMT]
        return img, label
    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class transformS2Data:
    def __init__(self, PMTstds, amp_thres=0.01, is_training=True,
                 amp_aug_level=0.1, gain_aug_level=0.01, width_aug_level=0.01, shift_aug_level=16):
        self.PMTstds     = PMTstds
        self.amp_thres   = amp_thres
        self.is_training = is_training
        self.amp_aug_level   = amp_aug_level
        self.gain_aug_level  = gain_aug_level
        self.width_aug_level = width_aug_level
        self.shift_aug_level = shift_aug_level
        if not isinstance(self.PMTstds, torch.Tensor):
            self.PMTstds = torch.from_numpy(self.PMTstds).float().view(1, 1, -1)
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        img = self._normalization(img)
        img = self._amplitude_augmentation(img)
        img = self._gain_augmentation(img)
        img = self._time_shift_augmentation(img)
        img = self._time_width_augmentation(img)
        img = self._amplitude_limit(img)
        return img
    def _normalization(self, img):
        if img.ndim == 2:
            img = img.unsqueeze(0) # [1, time, PMT]
        thres_mask = (img >= self.amp_thres).float()
        img = img*thres_mask
        img = torch.log1p(torch.clamp(img, min=0))
        img = img/(self.PMTstds + 1e-6)
        return img
    def _amplitude_limit(self, img):
        img = torch.clamp(img, 0, 10)
        return img
    def _amplitude_augmentation(self, img):
        if self.is_training is True:
            amp_aug = torch.randn_like(img)*self.amp_aug_level*img
            zero_mask = (img > 0).float()
            img = img + amp_aug*zero_mask
        return img
    def _gain_augmentation(self, img):
        if self.is_training is True:
            gain_aug = 1.0 + torch.randn(1).item()*self.gain_aug_level
            img = img*gain_aug
        return img
    def _time_width_augmentation(self, img):
        if self.is_training is True:
            width_aug = 1.0 + torch.randn(1).item()*self.width_aug_level
            width_aug = max(0.8, min(1.2, width_aug))
            width_orig = img.shape[1]
            width_new  = int(width_aug*width_orig)
            img = img.unsqueeze(0) 
            img = torch.nn.functional.interpolate(img, size=(width_new, img.shape[-1]), mode='bilinear', 
                                                  align_corners=False)
            img = img.squeeze(0)
            if width_new > width_orig:
                bin_start = (width_new - width_orig) // 2
                img = img[:, bin_start:bin_start+width_orig, :]
            elif width_new < width_orig:
                pad_left  = (width_orig - width_new) // 2
                pad_right = (width_orig - width_new) - pad_left
                img = torch.nn.functional.pad(img, (0, 0, pad_left, pad_right))
        return img
    def _time_shift_augmentation(self, img):
        if self.is_training is True:
            shift_aug = torch.randint(-self.shift_aug_level, self.shift_aug_level, (1,)).item()
            img = torch.roll(img, shifts=shift_aug, dims=1)
        return img
###############################################################################################################
def main():
    verbosity  = 2
    randomSeed = 11

    epochN     = 300
    batchSize  = 64
    learningRate  = 0.0005

    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)
    torch.backends.cudnn.deterministic = True
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
    dataDir = os.path.expanduser('/scratch/tmp/ylin3/dataset/S2data/')
    with open(dataDir+'Kr83m_sim_sample__10000_s2_data_long_per_channel_0pad_PMTstds.pkl', 'rb') as pickleFile:
        imageSTD = pickle.load(pickleFile)
    classes = ['Kr83m', 'fake']
    trainPaths0 = [dataDir+'Kr83m_10000_s2_data_long_per_channel_0pad_0.h5']
    testPaths0  = trainPaths0
    trainPaths1 = [dataDir+'fake_10000_s2_data_long_per_channel_0pad_0.h5']
    testPaths1  = trainPaths1
    trainData = lazyH5Data_2Darrays(trainPaths0, trainPaths1,
                                    class_names=classes, ratio_range=[0.0, 0.9], rand_seed=randomSeed,
                                    transform=transformS2Data(PMTstds=imageSTD, is_training=True))
    testData  = lazyH5Data_2Darrays(testPaths0, testPaths1,
                                    class_names=classes, ratio_range=[0.9, 1.0], rand_seed=randomSeed,
                                    transform=transformS2Data(PMTstds=imageSTD, is_training=False))
    loaderArgs = {'batch_size': batchSize}
    if GPUNAME == 'cuda':
        loaderArgs = {'batch_size': batchSize, 'num_workers': 8, 'pin_memory': True}
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, shuffle=True,  worker_init_fn=seed_worker,
                                              **loaderArgs)
    testLoader  = torch.utils.data.DataLoader(dataset=testData,  shuffle=False, worker_init_fn=seed_worker,
                                              **loaderArgs)
    if verbosity >= 1:
        print('trainPaths0:', trainPaths0)
        print('trainPaths1:', trainPaths1)
        print('testPaths0:', testPaths0)
        print('testPaths1:', testPaths1)
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






