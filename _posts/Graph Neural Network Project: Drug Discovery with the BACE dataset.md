### Implementation/Methodology

I initially started work on the dataset using Google Colabs as it has free GPU-access and a convenient notebook style format. I was able to perform preliminary EDA by loading in the data from a CSV file and drawing the molecules using RDKit's MolFromSmiles and MolToImage methods. However, loading the whole CSV file into a dataloader for the training was impractical and I chose to create a custom dataset using PyTorch Geometric's Dataset class. This offered several advantages:

- Although the dataset is small (only 1513 rows with a few columns of data) it was more efficient to load the CSV file once, perform the processing and save each processed molecule. Otherwise we would have to load and process the csv file each time we created the dataloader. 
- If in future work we decided to work with more data and the dataset was bigger than CPU memory, using a custom dataset would be more flexible for loading the dataset.
- A custom dataset is more flexible for working with non-tabular datatypes. 
- We would be able to perform any pre-transforms or transforms to the data while creating the dataset and only have to perform this once.  



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
