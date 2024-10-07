# Authors : 
* Souffir Ilan Meyer
* Derhy Avidan
* Aqua Gal

## Subject
In this project, our first mission was to run and adapt last year's project to work with our, we added tests to find if it's a known or unknown speaker, we also improve all codes of last year project.<br>
Our second and main mission was to built several models for audio classifications using a variety of clustering methods.
We are able to classify new audio files to known and unkown speakers:<br> 

1. RadiusNeighbors; We began by pre-processing the data using Librosa, to extract the mfcc features. Similar to k-means, but more suitable to classify unkown speakers, why excluding audio files that are too far from the existing means, and marking them as "unkown".<br>

2. DEC (Deep Embedding for Clustering) with CRNN; based on https://ieeexplore.ieee.org/document/9538747, where we replaced the Resnet with a CRNN. DEC uses a NN as an encoder for the audio files (i.e. extracting their features) before clustreiring them.<br>
Initially, we tried using the model decribed in the paper, with a Resnet as an encoder for our audio files, and later we also tried using the EfficientNet pre-trained model instead, however, we found that by using our own customized CRNN as the data encoder (to extract the features) we were able to improve upon the accuracy provided by the pre-trained models.<br>
By encoding with the pre-trained EfficientNet the accuracy reached around 60%. By encoding with our custom CRNN the accuracy reached 96%, and more clusters were formed.<br> 

The difference in their performances stems from the merits of the CRNN. On one hand, as an RNN, it effectively processes sequential input (in our case, audio), while on the other hand, as a CNN, it efficiently handles the frequencies and amplitudes of waves.<br>

3. DynMSC (Dynamic Medoid Silhouette Clustering); based on https://arxiv.org/pdf/2008.05171, and also https://python-kmedoids.readthedocs.io/en/latest/#dynmsc. This is a version of FasterMSC with automatic cluster number selection, that performs FasterMSC for a minimum k to the number of input medoids and returns the clustering with the highest Average Medoid Silhouette. In Silhouette clustering our goal is to minize the formula.<br>
<p align="center">
  <img align="center" width=30% src = "https://github.com/user-attachments/assets/fccceeec-509c-4b98-8bd5-557b4595a2d3"/>
</p>

## Before to run the project
you need to download the librispeech dataset, and then adapte the number of speakers you want for the tests.<br>
Notes : we use for the clustering part 100 speakers with each one have 30 audio files (4s/audio).

## parts of the project
### Improve of last year project:
* CustomeDataset.py
* check_pth_content.py
* cnn_model_wce.py
* constants.py
* sigproc.py
* test_100wav.py
* test_with_simple_audio.py
* training.py
* wav2vec2.py
* wav_reader.py
* 
### Clustering part:
* clustering_DEC_CRNN.py
* clustering_DEC_EfficientNet.py
* clustering_DEC_Wav2vec2.py
* clustering_RadiusNeighbors.py
* clustering_k-medoids_DYNMSC.py
* clustering_preProcess.py
* clustering_visualization.py
* view_model.py
