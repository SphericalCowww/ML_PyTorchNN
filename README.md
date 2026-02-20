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

    sudo apt install nvtop
    nvtop                        # for monitoring GPU
    sudo apt install psensor
    # check psensor App, for monitoring temperatures 
    # check also APP "NVIDIA X Server Settings" to adjust GPU settings

## Bullets:

  * for review: go through StatQuest in the references and z0...py to z3...py
  * dataset from MNIST, and cat&dog data set from online linked in the reference
  * set ``checkpointLoadPath = None`` to redo the training every time
  * for tracking, use TensorBoard on the recorded data, e.g., ``tensorboard --logdir yTransferModelTemplate/model1`` and go to the browser ``http://localhost:6006``

## References:
- StatQuest with Josh Starmer, concepts on neural networks in general (<a href="https://www.youtube.com/watch?v=CqOfi41LfDw">YouTube1</a>, <a href="https://www.youtube.com/watch?v=IN2XmBhILt4">YouTube2</a> are sufficient for the main idea, and <a href="https://www.youtube.com/watch?v=GKZoOHXGcLo">YouTube3</a>, <a href="https://www.youtube.com/watch?v=GKZoOHXGcLo">YouTube4</a> are for backpropagation)
- StatQuest with Josh Starmer, path towards transformer (<a href="https://www.youtube.com/watch?v=AsNTP8Kwu80">RNN</a> => <a href="https://www.youtube.com/watch?v=YCzL96nL7j0">LSTM</a> => <a href="https://www.youtube.com/watch?v=viZrOnJclY0">token (word) embedding</a> => <a href="https://www.youtube.com/watch?v=L8HKweZIOmg">seq2seq</a> => <a href="https://www.youtube.com/watch?v=PSs6nxngL6k">attention layer</a> => <a href="https://www.youtube.com/watch?v=zxQyTK8quyY">transformer</a>: token embedding (token: unit of sequential input); position embedding; self-attention layer with query, key, and value (<a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">QKV</a>); residual connection; <a href="https://www.youtube.com/watch?v=GDN649X_acE">encoder</a> / decoder; multilayer perceptron (fully connected layer at decoder))
- Patrick Loeber, Deep Learning With PyTorch - Full Course (<a href="https://www.youtube.com/watch?v=c36lUUr864M">YouTube</a>)
- Bhavik Jikadara, Cats and Dogs Classification Dataset (<a href="https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset">WebSite</a>)
- mildlyoverfitted, Vision Transformer in PyTorch (<a href="https://www.youtube.com/watch?v=ovB0ddFtzzA">YouTube</a>)
