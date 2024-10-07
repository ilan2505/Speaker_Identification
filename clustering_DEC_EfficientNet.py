import os
import numpy as np
import librosa
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def extract_mfcc(file_path, sr=22050, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc



def load_audio(root_dir):
    audioNlabels = []
    for i, speaker_dir in enumerate(os.listdir(root_dir)):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    audioNlabels.append([extract_mfcc(file_path), i])
    return audioNlabels



def EfficientNet_encoder(mfcc_features):
    # Assuming mfcc_features is of shape (n_mfcc, time_steps)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.transpose(mfcc_features, (0, 2, 1))  # Reorder dimensions if necessary
    
    # Create a placeholder image array
    # Adjust this based on how you want to process your features
    img = np.zeros((224, 224, 3))  # Placeholder image
    img = preprocess_input(img[None, ...])  # Preprocess for EfficientNet

    model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = model.predict(img)
    return features.flatten()



def classify_with_radius_neighbors(featuresNlabels, radius=1.0):
    encoded_features = []
    labels = []
    # Encode the features
    for mfcc_features, label in featuresNlabels:

        encoded = EfficientNet_encoder(mfcc_features)
        encoded_features.append(encoded)
        labels.append(label)
    
    # Convert to numpy arrays
    encoded_features = np.array(encoded_features)
    labels = np.array(labels)
    print('endocesc ', encoded_features.shape)

    clf = RadiusNeighborsClassifier(radius=radius)
    clf.fit(encoded_features, labels)

    # Predict the labels using the same features (to check clustering quality)
    predicted_labels = clf.predict(encoded_features)
    
    # Print cluster information
    unique_labels = np.unique(predicted_labels)
    print(unique_labels)
    for label in unique_labels:
        cluster_points = encoded_features[predicted_labels == label]
        print(f"Cluster {label} has {cluster_points.shape[0]} points.")

    # PCA for visualization
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(encoded_features)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('Cluster Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('dec_results.png')
    plt.show()
    
    # Evaluate accuracy of clustering
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return predicted_labels, accuracy



if __name__=='__main__':
    featuresNlabels = load_audio(root_dir='audio')
    predicted_labels, accuracy = classify_with_radius_neighbors(featuresNlabels)