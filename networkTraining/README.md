# Network Training

We train our network on a server running Linux system with 4 Nvidia GeForce GTX TITAN GPUs, and we support the *multiple GPUs* parallel-training technique. The training script is **ONLY** tested on Linux system.

Clone the repository and enter the network training part:

    git clone https://github.com/Enigma-li/Sketch2CAD.git
    cd networkTraining

There are four sub-folders under the training project root ***network*** folder.
* *libs* folder contains the custom training data decoder implemented in C++ and imported as custom ops in TensorFLow framework.
* *data* contains the instructions to download training, evaluation and testing data in TFRecords format.
* *script* folder contains the network building, data loading, training and testing scripts.
* *utils* folder contains the utility functions.


## Installation

To install the training script, please first check the system and packages requirements.

### System requirement

We list the TensorFlow and Python versions, as well as some required packages:

* Our TensorFlow version is ***1.3.0***, other versions are not tested
* Our Python version is ***3.6***, other versions are note tested
* To implement the custom data decoder and monitor the training process, we require some additional packages installed:
  * Zlib - `sudo apt-get install zlib1g-dev`
  * opencv-python - `pip install opencv-python`

Other packages could be installed via `pip` if needed.

💡 ***Suggestions***: ince we build the training script based on our previous [SketchCNN](https://github.com/Enigma-li/SketchCNN) project, thus we do not upgrade the Tensorflow version. We highly encourage users to use Docker to deploy the corresponding environment.

### Install

We first build the custom ops in *libs* folder and then configure the Linux system to compatible with the training script. You will see the file named `custom_dataDecoder.so` generated in `libs` folder after building the ops.

* Enter *libs* folder and build the ops. Remember to change the TensorFlow source path in `build.sh` based on your system configuration.
  > cd libs <br /> ./build.sh
* Add `export PYTHONPATH="/path_to_this_repository/network"` to your `.bashrc` file if the project root is not detected.
* Add `export CUDA_DEVICE_ORDER="PCI_BUS_ID"` to your `.bashrc` file to support multi-GPU specification.

## Usage
With the project installed and custom ops built successfully, now you could try the training and testing script.

Enter the `script` folder, you will see some files whose names are beginning with "train" or "test", these will be the training and testing scripts. We accept the console parameters for training and testing configuration, just type python with `-h/--help` command to see the usage, e.g., 

    $python train_addSub_reg.py -h
    $python test_addSub_reg.py -h 

### Training

We show one example below for addSub operator training:

    usage: train_addSub_reg.py [-h] --dbTrain DBTRAIN --dbEval DBEVAL --outDir
                               OUTDIR [--nb_gpus NB_GPUS] [--devices DEVICES]
                               [--ckpt CKPT] [--cnt CNT] [--rootFt ROOTFT]						 
                                                                       
    optional arguments:                                            
      -h, --help         	show this help message and exit
      --dbTrain DBTRAIN  	training dataset directory
      --dbEval DBEVAL    	evaluation dataset directory
      --outDir OUTDIR    	output directory
      --nb_gpus NB_GPUS  	GPU number
      --devices DEVICES  	GPU device indices
	  --ckpt CKPT        	checkpoint path
	  --cnt CNT          	continue training flag
      --rootFt ROOTFT    	root feature size

💡Here, 
* GPU number, GPU device indices have default value, modify them based on your system. 
* Output directory will be created automatically if does not exist. 
* Parameters, such as training data path should be specified by users.
* *Other training parameters are hardcoded at the very beginning of the script (**hyper_params**), you could change them to some values you want and it is easy to get the meaning of them.*

One typical training command for this will be:

    $python train_addSub_reg.py --dbTrain=../data/train --dbEval=../data/eval --outDir=../output/train_addSub

To monitor the training process, you could use the `TensorBoard` tool as shown below:

    $cd ../output/train_addSub
    $tensorboard --logdir=train
    
Now you could access the training in your Browser via: `http://localhost:6006`. Be happy to use the fancy tool. :)


### Testing
We show the testing process of addSub operator.

The checkpoints are written into the folder `../output/train_addSub/savedModel`. When the training converged (about `10` epochs), you could test the network and collect the training losses values and output maps (written to `EXR` image).

    usage: test_addSub_reg.py [-h] --dbTest DBTEST --outDir OUTDIR
                              [--devices DEVICES] [--ckpt CKPT] [--nbThr NBTHR]
							  [--rootFt ROOTFT]
                                                                             
    optional arguments:                                                      
      -h, --help         	show this help message and exit
      --dbTest DBTEST    	evaluation dataset directory
      --outDir OUTDIR    	output directory 
      --devices DEVICES  	GPU device indices
      --ckpt CKPT        	checkpoint folder                            
      --nbThr NBTHR      	n=Number of loading thread                                           
      --rootFt ROOTFT    	root feature size

💡Note that:
* when testing, we will write the graph definition out to one `.pbtxt` file, specify the name as you want
* Other parameters are hardcoded at the very beginning of the script, change them if needed.

One typical testing command will be:

    $python test_addSub_reg.py --cktDir=../output/train_addSub/savedModel --dbTest=../data/test --outDir=../output/test/test_addSub

You will get the log file containing the training loss for every data item and the outputting images under folder `../output/test/test_addSub/out_image`.

### Combine trained networks into one big network
Since we train five sub-networks, when use in practice, we would choose the operation regression network based on the classification network outputs. Then we provide the tool to combine all the subnetworks into one big network, so that within one forward pass, we can get all the interested results from networks.

Check the detailed usage by executing:
	
	$python combine_whole_graph.py -h
	

### Freeze network to use in C++ project

We could deploy the trained networks in C++ project to develop the interactive design tool. To use the network, we must first freeze it and prepare the configuration file (i.e., input, output nodes), we provide the tool to freeze the network in folder `utils`.

The command line parameters for this tool are:

    usage: freeze_graph_tool.py [-h] [--output_dir OUTPUT_DIR]
								[--ckpt_dir CKPT_DIR] [--in_gn IN_GN]
								[--out_gn OUT_GN] [--out_nodes OUT_NODES]
								[--devices DEVICE]								
                                                                         
    optional arguments:                                                
      -h, --help            	show this help message and exit
      --output_dir OUTPUT_DIR   output folder
      --ckpt_dir CKPT_DIR   	checkpoint folder
      --in_gn IN_GN         	input graph name
      --out_gn OUT_GN       	output graph name
      --out_nodes OUT_NODES     output node name
	  --devices DEVICE      	GPU device indices 

💡 ***Special NOTE***: to use the trained network in C++ project in Windows, you should compile and build the ***SAME*** TensorFlow version, see more details in *Data generation and network deployment* part.


