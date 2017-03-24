# Training materials for deep learning

## List of resources and use cases

### Caffe examples
    - Intro to caffe
    - 00_MNIST_linear
    - 01_MNIST_full_connect
    - 02_MNIST_Lenet

### Keras examples
    - Intro to Keras
    - 01_char_languaje_model
    - 02_sentiment_model
    - 99_utils

### Keras 2.0 examples
    - Intro to Keras
    - 01_char_languaje_model
    - 02_sentiment_model
    - 99_utils


### Tensorflow examples
    - 00 - Intro to tensorflow
    - 01 - Template session
    - Template_class dir
    - Tensorboard_example dir (based on https://gist.github.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1#file-mnist-py)
    - TEXT
        - Word tagging
        - Sentiment model with convolutions
    

### Torch examples (work in progress)
    - Torch basics 




## List of deep learning courses

Deep learning course of Google in Udacity
    https://www.udacity.com/course/deep-learning--ud730

Deep learning course of fast.ai
    http://course.fast.ai/

Stanford course Deep learning for Natural languaje processing
    http://cs224d.stanford.edu/

Stanford course Convolutional Neural Networks for Visual Recognition
    http://cs231n.stanford.edu/

Machine learning course of Stanford in Coursera 
    https://www.coursera.org/learn/machine-learning




## Instructions to create a Tensorflow and Keras environment in windows

Install anaconda

    https://repo.continuum.io/archive/Anaconda3-4.3.0.1-Windows-x86_64.exe

Create a Tensorflow 1.0 and keras 2.0 environment

    conda create -n tensorflow python=3.5
    activate tensorflow
    conda install -c conda-forge matplotlib=1.5.3
    pip install --upgrade tensorflow
    pip install keras
    pip install jupyter

