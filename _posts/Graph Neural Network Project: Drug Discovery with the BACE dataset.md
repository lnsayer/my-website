## Introduction/Theory


#### Models used: 
I produced results for four different model architectures, which are discussed more in the theory section. 

- GCN (Graph Convolutional Network) - The graph convolutional operator from the [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) paper.
- GAT (Graph Attention Network) - The graph attentional operator from the [Graph Attention Networks](https://arxiv.org/abs/1710.10903) paper
- GIN (Graph Isomorphism Network) - The graph isomorphism operator from the [How Powerful are Graph Neural Networks?](paperhttps://arxiv.org/abs/1810.00826) paper
- GraphConv - graph neural network operator from the [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) paper.


## Implementation/Methodology

I initially worked on the dataset using Google Colabs as it has free GPU-access and a convenient notebook style format. I was able to perform preliminary EDA by loading in the data from a CSV file. However, loading the whole CSV file into a dataloader for the training was impractical and I chose to create a custom dataset using PyTorch Geometric's Dataset class. This offered several advantages:

- Although the dataset was small (only 1513 rows with a few columns of data) it was more efficient to load the CSV file once, perform the processing and save each processed molecule. Otherwise we would have to load and process the csv file each time we created the dataloader. 
- If in future work we decided to work with more data and the dataset was bigger than CPU memory, using a custom dataset would be more flexible for loading the dataset.
- A custom dataset is more flexible for working with non-tabular datatypes. 
- We would be able to perform any pre-transforms or transforms to the data while creating the dataset and only have to perform this once.

Google Colabs was very useful at the beginning of the project but some issues arose as I streamlined the project and implemented more models:
- The notebook became very long with all the models, analysis and random cells used to understand what the code was doing.
- I had to install PyTorch Geometric, Deepchem and RDKit everytime I loaded the the notebook as these were not pre-installed in Colabs.
- I had to create the dataset and dataloaders each time I loaded the notebook. I managed to save the processed molecules in my Google Drive but loading these each session was longer than performing the processing and saving it in the memory of the session.
- GPU-access was limited with the free version, I paid for Colab Pro but it did not suffice for heavy-use.

I tried to modularise my code within Google Colabs using the magic command `%%writefile` which saved a cell as a script so that it could be imported by other cells. This turned out to be less efficient since when importing other scripts, their respective imported modules would have to be imported each time (took ~10 seconds) and this made it very time consuming to make small changes to code. 

As a result I tried to setup the scripts locally on my mac but it was difficult to install PyTorch Geometric and DeepChem. I decided to run a dual boot dual drive on my desktop such that I could use a Linux operating system alongside Windows. On my Linux operating system I installed CUDA for access to my 1060GTX 6GB GPU and cuDNN for accelerating this. It took considerable time to find compatible package versions for PyTorch, PyTorch Geometric, DeepChem and RDKit. 

I used Visual Studio Code as my coding editor and created documented, modularised scripts for the different functionalities of the project. I was finally able to save the processed data separately and load it in to create dataloaders very quickly. I was also able to save metrics (loss, accuracy and AUC) from training runs in a separate directory to view results whenever required. The modularised scripts can be viewed [here](https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python).

Two features unique features I implemented in my code:
- I created an early stopping protocol since each model architecture trained optimally in a different amount of time. This worked by calculating averages of the loss and AUC from the last ten epochs (e.g if the model was on epoch 53 the average would be calculated from epochs 43-52). The model's parameters were updated if the current moving average (from the test set) was better than all the previous moving averages (i.e if the moving average loss is lower than any previous moving average loss AND the moving average AUC is higher than any previous moving average AUC). This protocol also implemented patience which meant that a training run would wait for a certain number of epochs (e.g. 50) for the metrics to improve before the run was stopped. Therefore the model would always be finally saved 50 epochs before the total number of epochs. 
This was very useful as it prevented me from having to manually set the number of epochs to train for and also prevented a model from overfitting.
- I also saved a random set of indices with which to split the whole dataset into the training set and test set (into a 80-20% split). I used these whenever creating the dataloaders. I could have used a random manual seed but this was a more reliable method for producing the same training and test sets.

The BACE dataset includes its molecules in SMILES format which I was able to convert into Graph format using DeepChem's [MolGraphConvFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer) and then into a PyTorch Geometric graph. The number of features of each graph was 30 and the number of edge attributes were 11. Some of the node features were for example atom type and formal charge and some of the edge attributes were for example bond type and whether they were in the same ring. 

I trained four different model architectures, as described in the introduction: GCN, GAT, GIN and GraphConv. I performed very little hyperparameter tuning and as such my optimiser and loss functions remained unchanged. I used the Adam optimiser since it has adaptive learning rates and the binary cross entropy loss as the loss function. I mostly used a learning rate of 0.001 for Adam but for the GIN models I used 0.0001. 

For two of the model architectures, namely GIN and GAT, I was also able to incorporate the edge attributes of the graphs and I called these models GINE and GATE. 

Graph classification was obtained from the node embeddings by using a pooling method. My most used pooling method was the global mean pooling which calculated an average of the nodes' embedded features to produce a single graph embedding. I also briefly tried global max pooling which finds, for each feature of the nodes, the highest value amongst the nodes.

Most of my models had 3 layers with 128 hidden channels per layer. My models therefore had this number of parameters: 

- GCN model: 37,250
- GAT model:  346,754
- GIN model: 73,858
- GraphConv model: 104,066
- GATE model: 346,754
- GINE model: 107,498


These are quite small models and if I choose to tackle a larger dataset I would likely want to increase the model size. 


## Results

During training runs I saved three different metrics at each epoch to measure the performance of the model: the loss, accuracy and AUC of both the training set and test set. In these results we will use the GCN model as the baseline since it is the most simple and compare it with the GIN (with edge attributes), which was our best model. The metrics results for the GCN model can be seen in Fig. 1: 

Figure 1:
<img src="https://lnsayer.github.io/my-website/files/bace_dataset/gcn_loss_acc_auc_plot.png" alt="Untitled" style="height:auto;">

As can be seen in Fig. 1 the loss decreases for both the training set and test set over the epochs. The training set loss probably decreases more smoothly than the test set for several reasons: the training set is four times larger than the test set and this helps to average out fluctuations. Also the loss from the training set directly influences the change in the model parameters and this inherently smoothes the training loss. 

Although the test set loss continues to decrease, the other test set metrics stagnate and this is why the last saved model is at around 150. 

The training set accuracy continues to increase at 200 epochs while the test set accuracy plateaus, which indicates the model is overfitting. The AUC for both the training set and test set plateaus after around 100 epochs. 

These metric plots nicely represent the metric plots of the other models. The two GIN models (with and without edge attributes) however converge much faster. The metric plots for the GIN (with edge attributes) can be seen in Fig. 2. 

Figure 2:

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/gineconv_loss_acc_auc_plots.png" alt="Untitled" style="height:auto;">

The test set loss actually appears to increase after around 40 epochs and the accuracy appears to decrease as well which is classic overfitting. This model seems to be capable of fitting to the data much more effectively than the GCN models. This is supported by the fact that the learning rate was set to 0.0001 for the GIN models, 10x lower than that of the others. 
