# Training dataset and network deployment

In this part, there will be instructions for downloading data, checkpoint and sample code for deploy network in C++ applications.

## Training data

Details about data rendering, please refer to the paper in our [project page](http://geometry.cs.ucl.ac.uk/projects/2020/sketch2cad/). Totally, we generate about 400k data items for training and 100k data items for testing (with data augmentation), with size of about **21GB** and **4.3GB**.

Now we provide the Google drive link for downloading training datasets:

>[Training data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EnBYOgQdwdVDp9Z4FUoflyYBKwbBjliTsPPzP9BiNB1ITQ?e=cCsIhg)


## Trained network

We provide Google drive links for downloading the checkpoint and frozen network files of our full network:
>[Checkpoint](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EpGhY_1_mDJPqn0cBk-G2rMB4wpR79wMQg-aLcP7hCoZvw?e=GODso0) <br />
>[Frozen network](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/Eu-_9xhJPnpAtyYOJG3WRCgBHVflkgBR6qaXaC8UYiSiKQ?e=lOqz2y)

## Network Deployment

To deploy the trained network in C++ project in Windows, users must compile and build the TensorFlow libs and dlls from source using the ***SAME*** version as in network training stage. Then the source code named `trained_network.h` and `trained_network.cpp` provide a way to use the network in C++.

💡💡💡 ***Tips***:
* We provide the frozen network where you could find the sample configuration files containing the input/output nodes, channel numbers and so on, just try it.
* The first network forward pass would be time-consuming (about 2s on 1080Ti GPU) because of the initialization of GPU and CUDA settings. So after loading the network, please first execute the `warmup` step, all other forward passes after this `warmup` would be fast, i.e., 70ms.
* Compiling and building TensorFlow from source under Windows is time consuming (*over 2 hours*), we use **Visual Studio 2015** to build **TensorFlow 1.3**, which works for us, other configurations are **not tested**. 
* Download the pre-built Tensorflow 1.3 from the [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/ErhAHO3B_PtIl3r_4Y6rANsB6asMMI2RRjAXkm0qiZz8Fw?e=KLNd46).

