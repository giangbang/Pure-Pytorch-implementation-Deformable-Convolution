# Pure-Pytorch implementation of Deformable Convolution

Back in 2022, I had a personal project which used [Differentiable Binarization](https://arxiv.org/abs/1911.08947) to detect the ancient Chinese in images. After training the model, I wanted to deploy it to a server for demonstration purpose, using only CPU. However, the [original codebase](https://github.com/MhLiao/DB) of DB, which uses deformable conv as layers in the networks, did not support running on CPU. I delved into the codebase of Deformable conv and discovered that it was  written in cuda, so I wrote my own implementation using pure Pytorch, translated from the cuda code. 

This implementation is compatible with the original cuda version, so that after training with the cuda version, the trained weights can be loaded into this Pytorch version and the two versions will output exactly the same without additional tunning. I was able to run BD on my personal computer without a cuda device thanks to this.
<!-- The early version of this code, thanks to which I was able to run DB on my personal computer without a cuda device, is left dormant on kaggle for years.  -->

Fast forward two years later, and I still have not seen any implementation or repo that support running deformable conv on CPU, so I decided to make this code into a github repo.

## How to use
If a working Pytorch library is presented, then no installation is needed. Just copying `deform_conv.py` to your project should be fine. For example usage, have a look at `examples.py`.