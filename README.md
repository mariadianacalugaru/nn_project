# MICN: Multi-scale local and global context modeling for long-term time series forecasting

Maria Diana Calugaru (1893272)
Finizzi Tiziano (1915610)

The project implements the architecture designed in the paper [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://openreview.net/pdf?id=zt53IDUR1U).

### Description

It focuses on the task of long-term forecasting. The problem to be solved is to predict values for a future period $X_{t+1},X_{t+2},...,X_{t+T−1},X_{t+T}$, based on observations from a historical period $X_1,X_2,...,X_{t−1},X_t$.

The aim is to implement the forecasting task with a more efficient model that relies on convolutions instead of the existing self-attention mechanisms.

For implementation purposes, we used a dataset containing the electricity consumption of 321 customers recorded hourly from 2012 to 2014. We worked on 2 tasks: 
- $Multivariate\ predict\ Multivariate$: all the features of the input are observed in order to predict simultaneously all the features in output.
- $Multivariate\ predict\ Univariate:$ all the features of the input are observed in order to predict only a target features in output.

---
#### The overall model architecture
![](/plots/overall.png)

<br>

- $\textbf{Multi-scale hybrid decomposition block}$

This module returns the ***seasonal*** part and ***trend-cyclical*** part of the input data that will be predicted by two separate blocks.

<br>

- $\textbf{Trend-cyclical prediction block}$

This block predicts trend-cyclical information using the linear regression strategy on the trend-cyclical part of data previously extracted.

<br>

- $\textbf{Seasonal prediction block}$

![](./plots/seasonal.png)

This block performes 3 kinds of embeddings of the seasonal part: time features encoding, positional encoding and value embedding. 
Then there is a $\textit{Multi-scale isometric Convolution(MIC) Layer}$, using different branches (corresponding to different scale sizes used to model potentially different temporal patterns). On each branch, there is a local-global module that  extracts the local features and the global correlations of the sequence. The architecture of the local-global module is the following one:

![](/plots/local-global.png)

Local features represent the characteristics of a sequence over a small period $T$, and global correlations are the correlations exhibited between many periods $T_1, T_2,...T_{n−1},T_n$.
This block adopts one-dimensional convolution to extract local features and then propose isometric convolution to model the global correlations between each segment of the local features.

---
### Results on multivariate predict multivariate
Training loss on 6 epochs of 570 batches each.
![](/plots/training.png)

Testing on a random batch:

![](/plots/testing.png)

---
### Results on multivariate predict univariate

Training loss on 6 epochs of 570 batches each. On this task the training is not as stable as before because it has to learn a single feature that should summarize the real 321 features. However testing results show that the predictions are quite good. 

![](/plots/training2.png)

Testing on a random batch:

![](/plots/testing2.png)

---
### How to run the code
Install the needed packets to run the code with the command:
```
pip install torch numpy pandas scikit-learn matplotlib seaborn
```
In order to run the code, open the ```MICN.ipynb``` and run each cell sequentially. 
For testing purposes, we loaded the weights of the trained model in ```checkpoint.pth``` (for multivariate predict multivariate)

For the multivariate predict univariate task, the flag ```forecasting_type``` has to be set to 'MS' and the weights to be loaded are in ```MS_checkpoint```.