---
layout: post
title: Graph Neural Networks for Drug Discovery in Alzheimer's Disease
cover-img: /files/bace_dataset/bace_dataset_molecules.png
thumbnail-img: /files/bace_dataset/bace_dataset_molecules.png
share-img: /files/bace_dataset/bace_dataset_molecules.png
tags: [Graph Neural Networks, Classification, Drug Discovery, GCN, GAT, GINConv, GraphConv, PyTorch Geometric]
author: Louis Sayer
mathjax: true
---
 

Graph neural networks are an exciting new class of neural networks that are proving very effective in processing relational data. In this project I developed several graph neural networks to classify molecules as drugs in the BACE dataset, which is very useful in the drug discovery process, particularly in hit identification. I achieved results as high as those in the literature with an average AUC score of 0.881 for my top model, GINE, compared to 0.884 for MolXPT, from paperswithcode [1]. I identified which architectures were more effective for these data and wrote clean, documented, modularised reusable code which I could employ in further work. My code can be found on my [Github](https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python) [2].

## Project Objectives

My main objective for this project was to learn to utilise PyTorch on an end-to-end machine learning project which simulated a project in industry. This meant handling the data pipeline, defining the neural networks, creating training pipelines and visualising the results. I wanted to modularise my code so that it could be extended in future projects and make full use of the GPU, as would be done in industry. The results of the work were secondary to this but I still produced results as good as those in the literature.  

## Introduction/Theory

### GNN Overview 

Graph neural networks have recently emerged as a powerful class of deep learning artificial networks, which process data structured as graphs. Graph neural networks are effective in many different fields such as recommmendation systems, drug/protein discovery and fraud detection [3]. 

Graphs are non-euclidean data structures composed of nodes and edges, as can be seen in Fig.1 below. 

Figure 1: The nodes (shown as circles) and edges (lines between the circles) of a graph [4].

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/GraphTypes.png" alt="Untitled" style="height:auto;">

The nodes of a graph represent instances of something such as a person in a social network, a town in a country or an atom in a molecule. Edges represent the connections between these and can be directed, undirected and also weighted. For example, one person following another on Instagram (directed edge), a motorway between towns (directed and weighted) or an atomic bond (undirected). 

Graphs are very good at describing data with relationships and interactions. They are not that dissimilar to other data types and their corresponding neural networks. Convolutional layers in computer vision for example can be seen as acting on an image whose nodes are the pixels and only adjacent pixels are connected by edges in the graph.  

### Recent GNNs Applications 

GNNs have rapidly expanded into numerous fields, offering powerful ways for modeling complex data with interconnected structures. The breakthrough in GNNs came in 2016 with the introduction of GCNs (Graph Convolutional Networks) by Thomas Kipf and Max Welling [5] and since then many more effective architectures have been developed. Here are two recent examples of the great potential of GNNs. 


#### Weather Forecasting 

One fascinating application of GNNs is in weather forecasting. Last year Google Deepmind released GraphCast which is based on a graph neural network and was considered at the time the most accurate 10-day global weather forecasting system in the world [6] [3]. It could make accurate predictions very quickly - a 10 day forecast could be calculated in less than a minute.  Conventional approaches can take hours of computation in a supercomputer with hundreds of machines. This also made it much cheaper in energy efficiency - about 1000x. The architecture and some of the predictions can be seen in Fig. 2. 

Figure 2: The architecture is based on a Graph Neural Network with an Encoder-Processor-Decoder configuration [3]. The earth is modelled as an iteratively refined icosahedron (20 equilateral-triangular faces) with each grid block being mapped to node attributes (encoding). The processor performs message passing on these with 16 unshared GNN layers. The decoder maps these learned features back to predict the next state.  

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/graphcast_image.png" alt="Untitled" style="height:auto;">

#### Protein Design 

Another application is in protein design. The goal of this is to create proteins with desired properties which can be used for drugs, enzymes or materials. In 2023, BakerLab released RosettaFold Diffusion which uses a diffusion model to generate new proteins from noise [7] [3]. Diffusion models for image generation for example begin with images of pure static and gradually remove noise until a clear picture is formed [7]. The model is trained to recover corrupted (noised) protein structures and can produce unseen structures by denoising from a random noise input. The model operates via a special type of graph neural network and showed great improvements over its competitors across a broad range of problems. Fig. 3 shows a generated molecule and validated image. 

Figure 3: A symmetrical protein assembly (left), generated by RFDiffusion and a validating image (right) produced by single molecule electron microscopy [2]. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/GNN-RFDiffusion.png" alt="Untitled" style="height:auto;">

### GNN Theory 

There are several different types of graph neural network tasks and this project focusses on graph classification which involves predicting whether a molecule, which can be represented as a graph, is a potential drug [8]. As discussed before, molecules are described well by graphs as the nodes represent atoms and the edges represent the bonds between them. 

One popular GNN model (the graph convolutional network [5]) involves using a graph convolutional layer to share information between the nodes in the graph and then make a prediction based on information encapsulated by all the nodes. 

The nodes contain information about the atoms such as their atom type and formal charge. These are called their node features, which can be seen in Fig. 4.

Figure 4: Node features in a graph in which some are known and some unknown. All of the node features in our data are known [9]. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/new_node_features_in_a_graph.png" alt="Untitled" style="height:auto;">

The node features can be updated to take in information from their neighbouring nodes. This occurs through message passing, in which node features are updated by an aggregated function (e.g. mean) of all the neighbour nodes. Fig.5 shows message passing for a small graph. 

Figure 5: This is a two layer graph neural network which shows how message passing is performed on the target node A. In the second round of message passing node A will update its node features based on information aggregated from its neighbours B, C and D. B, C and D have updated their nodes in the first round of message passing from (A, C), (A, B, E, F) and (A) respectively. We can therefore see how more rounds of message passing share information through the graph more widely. Each round increases the reach of a node by 1 [10]. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/message_passing.png" alt="Untitled" style="height:auto;">

There are different aggregation functions to choose from. One popular aggregation function is the weighted mean, weighted by the degree of the neighbour nodes. The degree of a node is how many nodes it is connected to. The updated node embedding is then passed through a weight matrix/neural network to reduce its dimensionality. This process of aggregation and passing through a neural network is done several times as required. This process can be seen in Fig.5. I recommend watching this [great series](https://www.youtube.com/watch?v=OI0Jo-5d190&list=PLSgGvve8UweGx4_6hhrF3n4wpHf_RV76_) on GNNs to better understand the theory [11]. 

Figure 6: Two layers of a Graph Convolutional Network (GCN), in which the nodes' features are aggregated (averaged) and then passed through a neural network (or weight matrix) to reduce their dimensionality. The node dimensionality can be reduced to 1 which is a probability score (e.g. of being a viable drug or not) [12]. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/message_passing_gcn.png" alt="Untitled" style="height:auto;">

### Application to drug discovery 

Traditional drug discovery is a complex, expensive process that can take years or even decades and yet still has a low success rate. It involves identifying a "target" which is a protein/molecule involved in a disease whose activity needs to be altered to tackle the disease. A "hit" is a molecule which can potentially interact with the target to obtain the desired effect. After identifying the target and hits, many stages are required to refine the hits and test their safety and viability as a drug. AI can aid in all of these stages but with regards to our dataset, one in particular is hit identification. 

Traditionally, hit identification would involve high-throughput screening to test thousands or millions of compounds in a short time, which requires significant resources and has a low hit rate. AI can be used to speed up this process by predicting the interactions between the target and hits before lab testing. It can therefore detect new potential drugs from chemical libraries or test existing candidates to reduce the number of molecules which need to be tested. Many companies are using AI in this way to make drug discovery more efficient such as BenevolentAI and Exscientia, amongst many others [13] [14].

### Models used: 
I tested four different neural networks in this project. These models had roughly the same architectures, with the main difference being the convolutional layers employed. The different layers/networks are listed below:  


#### GCN (Graph Convolutional Network)

- The graph convolutional operator from [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) [5]

This convolutional layer updates a target node's embeddings by aggregating (weighted average in our case) the source nodes. Then, a learnable, layer specific weight matrix  is applied to the aggregation. The aggregation is weighted inversely to the degree node. This final step is done because source nodes with fewer connections  are more likely to be important (in the context of a social network, friends with fewer friends are likely to be more important connections than friends with lots of friends like celebrities or influencers). 

The PyTorch Geometric documentation for the [GCN layer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv) [15].

[Here](https://www.youtube.com/watch?v=2KRAOZIULzw&t=487s) is a fantastic, much more detailed explanation [12].


#### GAT (Graph Attention Network)

- The graph attentional operator from [Graph Attention Networks](https://arxiv.org/abs/1710.10903) [16]

This convolutional layer acts very similarly to the GCN layer however the layer-specific weight matrix is applied to both the source node and target node. Then, an attention mechanism is applied to these two nodes. The attention mechanism indicates the importance of a certain source node's features to a target node i.e. it gives more attention to certain nodes. In our case the attention mechanism is a single layer feedforward network. We can apply more than one attention head such that we can attend to different parts of the source nodes. For example, one head might give more attention to the atom type of the source node while another head may give more attention to the hydrogen bonding. 

The attention function creates coefficients for the source nodes so the new target node can be constructed from a weighted sum of the source nodes. A more detailed explanation can be found here [5].

The PyTorch Geometric documentation for the [GAT layer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv) [17].

#### GIN (Graph Isomorphism Network)
 
 - The graph isomorphism operator from [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) [18]

This network has a slightly different architecture to the others. It is similar to the GCN network however it does not normalise (by degree node) the features of the source nodes. It also includes a source's own feature vector in the summation, with a weighting such that it can control how much weight to give to itself compared to other target nodes. 

Secondly, the GIN uses a multi-layer perceptron after summing over the nodes which can capture more complex, non-linear relationships. This is instead of the linear weight matrix. GCNs may end up producing similar aggregated representations of nodes because of the normalisation and weight matrix. GINs ensure this is not the case with the absence of normalisation and the use of a MLP. This makes the graph more injective such that different node neighbourhoods can be better distinguished. An injective function produces distinct results for distinct inputs. In the context of this convolutional layer it means that the aggregation of neighbourhood nodes will always be distinct, which is not always the result for other convolutional layers. As we will see, this makes the graph classification much more effective. 

The PyTorch Geometric documentation for the [GIN layer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv) [19].

#### GraphConv

- The graph neural network operator from [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) [20]

The GraphConv layer uses separate layer-specific weight matrices for the target node and source nodes, similar to the GIN layer. This controls how much a node's own features will determine its updated features. The features from other nodes are not normalised either and makes the graph more injective. Edge weights can also be incorporated to weight the features from a source node, however we used the default weight of 1 for every node in this project. 

The PyTorch Geometric documentation for the [GraphConv layer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv) [21].


### BACE Dataset 

The dataset used in this project is the BACE dataset, from Deepchem [22]. It features 1513 molecules with qualitative (binary label) binding results for Beta-secretase 1 (BACE-1) in humans. Beta secretase 1 is involved in pathways that create amyloid beta peptides, which are the main component of the amyloid plaques found in the brains of people with Alzheimer's disease [23]. In theory drugs which block this enzyme could prevent the build-up of beta amyloid peptides which form the amyloid plaques and therefore help slow or stop Alzheimer's disease. Unfortunately, in practice it appears that drugs inhibiting BACE-1 are ineffective. Pharmaceutical companies such as Merck and Co, Eli Lilly and Co and Astrazeneca have all halted trials for BACE-1 inhibitors after lack of clinical benefits to patients [24], [25], [26].

All data in the BACE dataset are experimental values reported in the scientific literature over the past decade. The molecules are formatted as SMILES, and some of these can be seen in Fig. 7. The molecules are quite small, with an average of 34 atoms per molecule.

Figure 7: Three molecules in the BACE dataset, shown in their SMILES notation and respective molecule skeletal formula, drawn by RDKit's Draw.MolToImage. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/bace_dataset_molecules.png" alt="Untitled" style="height:auto;">


## Implementation/Methodology

### Google Colabs

I initially worked on the dataset using Google Colabs as it has free GPU-access and a convenient notebook style format [27]. I was able to perform preliminary EDA by loading in the data from a CSV file. However, loading the whole CSV file into a dataloader for the training was impractical and I chose to create a custom dataset using PyTorch Geometric's Dataset class. This offered several advantages:

- Although the dataset was small (only 1513 rows with a few columns of data) it was more efficient to load the CSV file once, perform the processing and save each processed molecule. Otherwise we would have to load and process the csv file each time we created the dataloader. 
- If in future work we decided to work with more data and the dataset was bigger than CPU memory, using a custom dataset would be more flexible for loading.
- A custom dataset is more flexible for working with non-tabular datatypes. 
- We would be able to perform any pre-transforms or transforms to the data while creating the dataset and only have to perform this once.

Google Colabs was very useful at the beginning of the project but some issues arose as I streamlined the project and implemented more models:
- The notebook became very long with all the models, analysis and random cells used to understand what the code was doing.
- I had to install PyTorch Geometric, Deepchem and RDKit everytime I loaded the notebook as these were not pre-installed in Colabs.
- I had to create the dataset and dataloaders each time I loaded the notebook. I managed to save the processed molecules in my Google Drive but loading these each session was longer than performing the processing and saving it in the memory of the session.
- GPU-access was limited with the free version, I paid for Colab Pro but it did not suffice for heavy-use.

I tried to modularise my code within Google Colabs using the magic command `%%writefile` which saved a cell as a script so that it could be imported by other cells [28]. This turned out to be less efficient since when importing other scripts, their respective imported modules would have to be imported each time (took ~10 seconds) and this made it very time consuming to make small changes to code. 

### Local Work

As a result I tried to setup the scripts locally on my mac but it was difficult to install PyTorch Geometric and DeepChem. I decided to run a dual boot dual drive on my desktop such that I could use a Linux operating system alongside Windows. On my Linux operating system I installed CUDA for access to my 1060GTX 6GB GPU and cuDNN for accelerating this. It took considerable time to find compatible package versions for PyTorch, PyTorch Geometric, DeepChem and RDKit. 

I used Visual Studio Code as my coding editor and created documented, modularised scripts for the different functionalities of the project. I was finally able to save the processed data separately and load it in to create dataloaders very quickly. I was also able to save metrics (loss, accuracy and AUC) from training runs in a separate directory to view results whenever required. The modularised scripts can be viewed [here](https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python) [29].

### Technical Details 

Two unique features I implemented in my code:
- I created an early stopping protocol since each model architecture trained optimally in a different amount of time. This worked by calculating averages of the loss and AUC from the last ten epochs (e.g if the model was on epoch 53 the average would be calculated from epochs 44-53). The model's parameters were updated if the current moving average (from the test set) was better than all the previous moving averages (i.e if the moving average loss is lower than any previous moving average loss AND the moving average AUC is higher than any previous moving average AUC). This protocol also implemented a patience parameter which meant that a training run would wait for a certain number of epochs (e.g. 50) for the metrics to improve before the run was stopped. Therefore the model would always be finally saved 50 epochs before the total number of epochs trained for. 
This was very useful as it prevented me from having to manually set the number of epochs and also prevented a model from overfitting.
- I saved a random set of indices with which to split the whole dataset into the training set and test set (80-20% split). I used these whenever creating the dataloaders. I could have used a random manual seed but this was a more reliable method for producing the same training and test sets.

The BACE dataset includes its molecules in SMILES format which I was able to convert into Graph format using DeepChem's [MolGraphConvFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer) [30] and then into a PyTorch Geometric graph. The number of features of each graph was 30 and the number of edge attributes were 11. Some of the node features were for example atom type and formal charge and some of the edge attributes were for example bond type and whether they were in the same ring. 

I trained four different models, as described in the introduction: GCN, GAT, GIN and GraphConv. I performed very little hyperparameter tuning and as such my optimiser and loss functions remained unchanged. I used the Adam optimiser since it has adaptive learning rates and the binary cross entropy loss as the loss function. I mostly used a learning rate of 0.001 for Adam but for the GIN models I used 0.0001. The GIN models were probably overfitting with a learning rate of 0.001 because I omitted a dropout layer in these models. 

For two of the model architectures (GIN and GAT) I was also able to incorporate the edge attributes of the graphs and I called these models GINE and GATE. I hoped that the prediction quality would improve with this further information. 

Graph classification was obtained from the node embeddings by using a pooling method. I used global mean pooling the most which calculated an average of the nodes' embedded features to produce a single graph embedding. I also briefly tried global max pooling which finds, for each feature of the nodes, the highest value amongst the nodes. However, this pooling method performed slightly worse in the metrics so I stopped using it. 

### Model specifications

The architectures of the GCN, GAT, GraphConv and GATE models were as follows: 
- Convolutional layer
- Leaky ReLU
- Convolutional Layer
- Leaky ReLU
- Convolutional Layer
- Pooling Layer
- Dropout
- Linear Layer
- Softmax 

Therefore these models had 3 layers, each with 128 hidden channels per layer. 

The GIN and GINE models were structured slightly differently:
- Convolutional layer
- ReLU
- Convolutional Layer
- ReLU
- Convolutional Layer
- Pooling Layer
- MLP
- Softmax

The MLPs used in the convolutional layers had 3 layers each and 128 hidden channels. I did not include a dropout layer for the GIN(E) models which was an oversight as this would have helped to lessen the overfitting. In hindsight I should have used more similar methods such as the activation function (ReLU vs Leaky ReLU) and the inclusion of a dropout layer. 

The exact structures of my models can be found in my code on Github [29].

My models had these numbers of parameters: 

- GCN model: 37,250
- GAT model:  346,754
- GIN model: 73,858
- GraphConv model: 104,066
- GATE model: 346,754
- GINE model: 107,498

These are quite small models and if I choose to tackle a larger dataset I would likely want to increase the model size. 

## Results and Discussion 

### Loss, Accuracy and AUC curves 

During training runs I saved three different metrics at each epoch to measure the performance of the model: the loss, accuracy and AUC of both the training set and test set. I performed five repeats for each model and all metrics are an average of these repeats. In the following plots we will use the GCN model as the baseline since it is the simplest model and compare it with the GINE, the best model. The metrics results for the GCN model can be seen in Fig. 8: 

Figure 8: Three different metrics as a function of epoch for a single GCN model training run. The test set results are shown in orange and the training set results are shown in blue. The model's parameters are saved at epochs with a grey dashed line.  
<img src="https://lnsayer.github.io/my-website/files/bace_dataset/gcn_loss_acc_auc_plot.png" alt="Untitled" style="height:auto;">

As can be seen in Fig. 8 the loss decreases for both the training set and test set over epochs. The training set loss decreases more smoothly than the test set for several reasons: the loss from the training set directly influences the change in the model parameters and this inherently smoothes the training loss. Also, the training set is four times larger than the test set and this helps to average out fluctuations. 

Although the test set loss continues to decrease, the AUC metric stagnates and this is why the last saved model is at around 150. 

The training set accuracy continues to increase at 200 epochs while the test set accuracy plateaus, which indicates the model is overfitting. The AUC for both the training set and test set plateau after around 100 epochs. 

These metric plots nicely represent the metric plots of the other models. The two GIN models (with and without edge attributes) however converge much faster. The metric plots for the GINE can be seen in Fig. 9. 

Figure 9: Three different metrics as a function of epoch for a single GINE model training run. The test set results are shown in orange and the training set results are shown in blue. The model's parameters are saved at epochs with a grey dashed line.  

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/gineconv_loss_acc_auc_plots.png" alt="Untitled" style="height:auto;">

The test set loss actually appears to increase after around 40 epochs and the accuracy appears to decrease as well which is classic overfitting. This model seems to be capable of fitting to the data much more effectively than the GCN models. This is supported by the fact that the learning rate was set to 0.0001 for the GIN models, 10x lower than that of the others. The overfitting likely comes from the omission of a dropout layer in these models, which was an oversight. If we had included this we could have expected less overfitting and also the test set loss to be smoother as the generalisability of the model would improve.  

### Average metric scores

I checked that the above metric curves (specifically the loss) were reasonable for each set of training runs, i.e. they were not overfitting or underfitting. After this I calculated the average results of the five runs. This was done by saving the models' state dictionaries during training, loading the saved parameters and then making predictions on the test set, which were used to calculate the following metrics in Table 1. 

Table 1: Five different metrics for each of the four different GNN architectures. Two more models were trained with the edge attributes. These are averages of five repeats. The precision and recall are calculated for the positive class.  
| Metric | GCN | GATE | GAT | GraphConv | GIN	 | GINE |
| ------- | ---- | ------- | ------- | ------- | ---- | -------|
| AUROC | 0.815 | 0.820 | 0.836 | 0.862 | 0.877	 | 0.881 |
| Precision | 0.752 | 0.764 | 0.765 | 0.784 | 0.802	 | 0.811|
| Recall | 0.732 | 0.754 | 0.785 | 0.793 | 0.815 | 0.796 |
| Accuracy | 0.762 | 0.776 | 0.786 | 0.800 | 0.819	 | 0.817 |
| F-1 Score | 0.761 | 0.775 | 0.784 | 0.800 | 0.819	 | 0.816|

| Metric | GCN | GATE | GIN	 | GINE |
| ------- | ---- | ------ | ---- | -------|
| AUROC | 0.815 | 0.820 |0.877	 | 0.881 |
| Precision | 0.752 | 0.764 | 0.802	 | 0.811|
| Recall | 0.732 | 0.754 |0.815 | 0.796 |
| Accuracy | 0.762 | 0.776 |  0.819	 | 0.817 |
| F-1 Score | 0.761 | 0.775 | 0.819	 | 0.816|



| Model | Default hyperparameters | Best hyperparameters |
| ------ | ----------- | ---|
| Age imputed with median | 0.802 | 0.816 |
| Age imputed with KNN    | 0.800 | 0.829 |
| Age imputed with KNN along with test data  | 0.805 | 0.826 |

As we can see from Table 1 the best performing model over different thresholds was GINE with an AUROC value of 0.881. This is very comparable to the top results in the literature, as reported in PapersWithCode. The highest AUC score in [PapersWithCode](https://paperswithcode.com/sota/molecular-property-prediction-on-bace-1) is 0.884 from MolXPT [1].

The GINE model also has the highest precision score of 0.811. Precision is more important in the latter stages of drug discovery (lead optimisation, preclinical, clinical trials) where the costs and consequences of advancing a false positive are much higher. The GIN model had the best recall score of 0.815. Recall is more important in the early stages of drug discovery (hit discovery, target identification) when it is crucial to cast a wide net and identify all candidates. We would not want to miss potentially valuable compounds. 

The GraphConv models are not far behind the GIN models, however the GINE model took on average 2.25x fewer epochs to converge than GraphConv, corresponding to roughly 2.25x less time. It is interesting to see that the GATE models performed worse than the GAT models. This may be because the GAT models have a lot of parameters (at least compared to the GIN models) and they might be overfitting on the training data. It could also be that the edge attributes are irrelevant and just providing noise, however the GINE model would not perform better if that were the case. 

### Confusion Matrices 

We can see where the models are making incorrect predictions by looking at their confusion matrices. Fig. 10 and Fig. 11 show the confusion matrices for the average GCN and GINE models respectively (average of five repeats).

Figure 10: The confusion matrix for the GCN models (an average of the five repeats). 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/avg_confusion_matrix_gcn_conv.png" alt="Untitled" style="height:auto;">

Figure 11: The confusion matrix for the GINE models (an average of the five repeats). 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/new_avg_confusion_matrix_gine_conv.png" alt="Untitled" style="height:auto;">

We can see from the confusion matrices that the two models make incorrect predictions in the same areas. The proportion of false positives between the GCN and GINE models are on average 1.26 (33.6 : 26.6) and the proportion of false negatives between them is on average 1.28 (37.2 : 29.0). The GINE model is generally better overall, although not in any specific areas. 

### Classification Threshold

I also wanted to look at the sensitivity between precision/recall and the classification threshold. Figs. 12 and 13 show the precision and recall as a function of the threshold for the GCN models and GINE models respectively (average of five repeats). 

Figure 12: Precision and recall as a function of the classifcation threshold for the GCN models (average of the five). The precision is shown in blue and the recall in orange. The metrics are calculated for the positive class. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/precision_recall_threshold_plot_gcn.png" alt="Untitled" style="height:auto;">

Figure 13: Precision and recall as a function of the classifcation threshold for the GINE models (average of the five). The precision is shown in blue and the recall in orange. The metrics are calculated for the positive class. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/precision_recall_threshold_plot_gine.png" alt="Untitled" style="height:auto;">

The GCN model's curve shows a moderately even trade-off between precision and recall with the precision taking over the recall at 0.4 (near to 0.5). The recall for the GINE model is much more sensitive and weaker overall. It is never higher than the precision, even at low thresholds and decreases quickly. This indicates the model is conservative at lower thresholds and becomes more selective with a higher threshold. The GCN model on the other hand is more likely to predict positives. The choice of the threshold is very much dependent on the requirements of the model application and it is very useful to see how these models perform. We can plot the precision and recall curves of the two models on the same plot for direct comparison as can be seen in Fig. 14. 

Figure 14: Precision as a function of recall. The GINE model results are shown in blue line and the GCN model results are shown in the orange. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/precision_recall_gine_gcn.png" alt="Untitled" style="height:auto;">

This shows that the GINE model is much more sensitive as it spans a wider range in both the recall and precision for the same classification thresholds. It also performs much better with the precision being higher at every recall. 

### Graph embedding visualisations with t-SNE

One final way to visualise the models' predictions is by showing dimensionally-reduced representations of the graph embeddings. I used a method called t-SNE (t-distributed stochastic neighbour embedding) to reduce the dimensionality of the final graph embeddings from 128 (the number of hidden channels) to 2 to see how the positive/negative classes cluster. I used a GINE model for this and we can see the plot in Fig. 15.

Figure 15: The dimensionally-reduced graph embeddings for all the positive and negative (predictive) cases of the 1513 molecules in the BACE dataset. The graph embeddings were reduced using t-SNE. The positive predictions are shown in blue and the negative predictions are shown in orange. 

<img src="https://lnsayer.github.io/my-website/files/bace_dataset/t-sne_visualisation_graph_embeddings.png" alt="Untitled" style="height:auto;">

It is clear from Fig. 15 that the model is able to differentiate between the two groups of graphs (positive and negative). It is normal that there are mixed regions for two reasons: the model may not be embedding the graphs perfectly and also t-SNE has some degree dimensionality reduction. The separation might be more distinct in higher dimensions however overall this is good clustering. 

## Conclusion 

I used Python and PyTorch to conduct an end-to-end machine learning project to classify small molecules as drugs with the BACE dataset. I achieved good results, with my top model GINE achieving an average AUC score of 0.881 which is very comparable to the top score in the literature of 0.884 by MolXPT [1]. My work shows the dominance of certain neural network architectures (GINE) over others (GCN/GAT) as my top models achieved much better results, with fewer parameters, in less training time. I created custom datasets and dataloaders, organised training and testing runs and implemented GNN models all in the framework of modularised, reusable code. Overall I am very proud of my work and have learnt a lot in my first substantial PyTorch project. 

This work could be further developed for use in drug discovery projects, either at the beginning of a project when it is important to identify all possible candidates or at the end when false positives become more costly. This ties in to future work in which I would like to reuse my code to tackle a larger, more complex dataset which is more realistic. Neural network architecture/models are the focus in academia however in industry, data is more important as high customisation is required. I would like to focus more on this to gain relevant skills for future industry work. 

[1] Molecular Property Prediction on BACE, paperswithcode, https://paperswithcode.com/sota/molecular-property-prediction-on-bace-1

[2] https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python

[3] AI trends in 2024: Graph Neural Networks, Marco Ramponi, https://www.assemblyai.com/blog/ai-trends-graph-neural-networks/

[4] Graph Theory Using Python – Introduction And Implementation, Active State, https://www.activestate.com/blog/graph-theory-using-python-introduction-and-implementation/

[5] Semi-Supervised Classification with Graph Convolutional Networks, Thomas N. Kipf, Max Welling, https://arxiv.org/abs/1609.02907

[6] GraphCast: AI model for faster and more accurate global weather forecasting, Remi Lam, https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/

[7] RFdiffusion: A generative model for protein design, https://www.bakerlab.org/2023/07/11/diffusion-model-for-protein-design/

[8] A Comprehensive Introduction to Graph Neural Networks (GNNs), Datacamp, https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial

[9] Michael Bronstein, Feature Propagation is a simple and surprisingly efficient solution for learning on graphs with missing node features, https://towardsdatascience.com/learning-on-graphs-with-missing-features-dd34be61b06

[10] Omar Hussein, Graph Neural Networks Series | Part 4 |The GNNs, Message Passing & Over-smoothing, https://medium.com/the-modern-scientist/graph-neural-networks-series-part-4-the-gnns-message-passing-over-smoothing-e77ffee523cc

[11] Intro to Graphs and Label Propagation Algorithm in Machine Learning, WelcomeAIOverlords https://www.youtube.com/watch?v=OI0Jo-5d190&list=PLSgGvve8UweGx4_6hhrF3n4wpHf_RV76_

[12] Graph Convolutional Networks (GCNs) made simple
, WelcomeAIOverlords, https://www.youtube.com/watch?v=2KRAOZIULzw&t=487s

[13] We used our BenAI Engine to identify a leading COVID-19 treatment, which is now FDA approved, BenevolentAI, https://www.benevolent.com/about-us/sustainability/covid-19/ 

[14] Exscientia announces second molecule created using AI from Sumitomo Dainippon Pharma collaboration to enter Phase 1 clinical trial, Exscientia, https://investors.exscientia.ai/press-releases/press-release-details/2021/Exscientia-announces-second-molecule-created-using-AI-from-Sumitomo-Dainippon-Pharma-collaboration-to-enter-Phase-1-clinical-trial/default.aspx 

[15] conv.GCNConv, Torch Geometric, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv

[16] Graph Attention Networks, Petar Veličković et al. , https://arxiv.org/abs/1710.10903

[17] conv.GATConv, Torch Geometric, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv

[18] How Powerful are Graph Neural Networks?, Keyulu Xu et al., https://arxiv.org/abs/1810.00826

[19] conv.GINConv, Torch Geometric, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html#torch_geometric.nn.conv.GINConv

[20] Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
, Christopher Morris et al., https://arxiv.org/abs/1810.02244

[21] conv.GraphConv, Torch Geometric, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html

[22] MoleculeNet, DeepChem, https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html

[23] Beta-secretase 1, Wikipedia, https://en.wikipedia.org/wiki/Beta-secretase_1

[24] Merck Stops Phase 3 Trial of Verubecestat in Early Alzheimer’s Patients Amid Concerns Over Benefit, Alzheimer's Today, https://alzheimersnewstoday.com/news/merck-stops-phase-3-study-verubecestat-in-early-alzheimers-patients/?cn-reloaded=1

[25] Lilly Provides Update on A4 Study of Solanezumab for Preclinical Alzheimer's Disease, Lily Investors, https://investor.lilly.com/news-releases/news-release-details/lilly-provides-update-a4-study-solanezumab-preclinical

[26] Update on Phase III clinical trials of lanabecestat for Alzheimer’s disease, Astrazeneca, https://www.astrazeneca.com/media-centre/press-releases/2018/update-on-phase-iii-clinical-trials-of-lanabecestat-for-alzheimers-disease-12062018.html#

[27] My initial colabs notebook graph_classification_bace_dataset.ipynb, https://github.com/lnsayer/drug_discovery_with_bace_dataset/blob/main/graph_classification_bace_dataset.ipynb

[28] My modularised colab notebook, https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_ipynb

[29] My modularised Python scripts folder, https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python

[30] MolGraphConvFeaturiser, Deepchem, https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer
