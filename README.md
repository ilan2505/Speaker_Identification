In this project, we had built several models for audio classifications using a variety of clustering methods.

We are able to classify new audio files to known and unkown speakers: 

1. RadiusNeighbors; We began by pre-processing the data using Librosa, to extract the mfcc features. Similar to k-means, but more suitable to classify unkown speakers, why excluding audio files that are too far from the existing means, and marking them as "unkown".

2. DEC (Deep Embedding for Clustering) with CRNN; based on https://ieeexplore.ieee.org/document/9538747, where we replaced the Resnet with a CRNN. DEC uses a NN as an encoder for the audio files (i.e. extracting their features) before clustreiring them.
Initially, we tried using the model decribed in the paper, with a Resnet as an encoder for our audio files, and later we also tried using the EfficientNet pre-trained model instead, however, we found that by using our own customized CRNN as the data encoder (to extract the features) we were able to improve upon the accuracy provided by the pre-trained models.
By encoding with the pre-trained EfficientNet the accuracy reached around 60%. By encoding with our custom CRNN the accuracy reached 96%, and more clusters were formed. 

The difference in their performances stems from the merits of the CRNN. On one hand, as an RNN, it effectively processes sequential input (in our case, audio), while on the other hand, as a CNN, it efficiently handles the frequencies and amplitudes of waves.

3. DynMSC (Dynamic Medoid Silhouette Clustering); based on https://arxiv.org/pdf/2008.05171, and also https://python-kmedoids.readthedocs.io/en/latest/#dynmsc. This is a version of FasterMSC with automatic cluster number selection, that performs FasterMSC for a minimum k to the number of input medoids and returns the clustering with the highest Average Medoid Silhouette. In Silhouette clustering our goal is to minize the formula. 
![alt text](Silhouette_formula.png)

Notes: initially we considered using HDBSCAN as well, but it was less convenient for classifying unkown speakers (even though it is possible, by creating a new cluster to represent the unkown speakers). We did use HDBSCAN to visualize the data and the clusters after PCA reduction.