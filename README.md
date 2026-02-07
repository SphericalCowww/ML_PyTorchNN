# PyTorch Neural Network

## Keywords:
- understanding CNN diagram dimensions: thickness comes from the number of different filters/kernels; no reduction in layer widthxheight due to zero-padding. The layer widthxheight only reduces in size after of max pooling (<a href="https://stackoverflow.com/questions/65554032/understanding-convolutional-layers-shapes">stackoverflow</a>)
- bias-variance-tradeoff versus precision-recall-tradeoff (<a href="https://stackoverflow.com/questions/65554032/understanding-convolutional-layers-shapes">stackoverflow</a>)

## Setting up CUDA for PyTorch:

    conda create -n cuda-env python=3.10
    conda activate cuda_env
    conda env list
    cat /proc/driver/nvidia/version # checking GPU driver version 
    nvidia-smi # verifying that the drive is running and record required cuda version
    conda install -c nvidia cuda-toolkit
    which nvcc
    nvcc --version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128     python
    >>> import torch
    >>> print("CUDA available:", torch.cuda.is_available())
    >>> print("PyTorch CUDA version:", torch.version.cuda)
    >>> print("GPU device count:", torch.cuda.device_count())
    >>> print("Device name:", torch.cuda.get_device_name(0))

## Bullets:

  * for review: go through StatQuest in the references and z0...py to z3...py
  * set ``checkpointLoadPath = None`` to redo the training every time
  * for tracking, use TensorBoard on the recorded data, e.g., ``tensorboard --logdir yTransferModelTemplate/model1`` and go to the browser ``http://localhost:6006``

## References:
- StatQuest with Josh Starmer, concepts on neural networks in general (<a href="https://www.youtube.com/watch?v=CqOfi41LfDw">Youtube1</a>, <a href="https://www.youtube.com/watch?v=IN2XmBhILt4">Youtube2</a> are sufficient for the main idea, and <a href="https://www.youtube.com/watch?v=GKZoOHXGcLo">Youtube3</a>, <a href="https://www.youtube.com/watch?v=GKZoOHXGcLo">Youtube4</a> are for backpropagation)
- Patrick Loeber, Deep Learning With PyTorch - Full Course, (<a href="https://www.youtube.com/watch?v=c36lUUr864M">Youtube</a>)
