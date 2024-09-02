### Implementation/Methodology

I initially worked on the dataset using Google Colabs as it has free GPU-access and a convenient notebook style format. I was able to perform preliminary EDA by loading in the data from a CSV file. However, loading the whole CSV file into a dataloader for the training was impractical and I chose to create a custom dataset using PyTorch Geometric's Dataset class. This offered several advantages:

- Although the dataset was small (only 1513 rows with a few columns of data) it was more efficient to load the CSV file once, perform the processing and save each processed molecule. Otherwise we would have to load and process the csv file each time we created the dataloader. 
- If in future work we decided to work with more data and the dataset was bigger than CPU memory, using a custom dataset would be more flexible for loading the dataset.
- A custom dataset is more flexible for working with non-tabular datatypes. 
- We would be able to perform any pre-transforms or transforms to the data while creating the dataset and only have to perform this once.

There were some issues with using Google Colabs as I tried to streamline the project and implemented more models. 
- The notebook became very long with all the models, analysis and random cells used to understand what the code was doing.
- I had to install PyTorch Geometric, Deepchem and RDKit everytime I loaded the the notebook as these were not pre-installed in Colabs.
- I had to create the dataset and dataloaders each time I loaded the notebook. I managed to save the processed molecules in my Google Drive but loading these each session from Google drive was longer than performing the processing and saving it in the memory of the session.
- GPU-access was limited with the free version, I paid for Colab Pro but it did not suffice for heavy-use.

I tried to modularise my code within Google Colabs using the magic command `%%writefile` which saved a cell as a script so that it could be imported by other cells. This turned out to be less efficient since when importing other scripts, their respective imported modules would have to be imported each time (took ~10 seconds) and this made it very time consuming to make small changes to code. 

As a result I tried to setup the scripts locally on my mac but it proved very difficult to install PyTorch Geometric and DeepChem. I decided to run a dual-boot dual drive on my desktop such that I could use a Linux operating system alongside Windows. On my Linux operating system I installed CUDA for access to my 1060GTX 6GB GPU and cuDNN for accelerating this. It took considerable time to find compatible package versions for PyTorch, PyTorch Geometric, DeepChem and RDKit. 

I used Visual Studio Code as my coding editor and created documented, modularised scripts with the different 


### Results

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
