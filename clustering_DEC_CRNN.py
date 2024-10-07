import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import RadiusNeighborsClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, UpSampling1D, Reshape
from clustering_preProcess import *



def crnn_encoder(input_shape, latent_dim=256):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    
    encoded = Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    x = Dense(input_shape[0] // 4 * 128, activation='relu')(encoded)
    x = Reshape((input_shape[0] // 4, 128))(x)
    
    x = LSTM(128, return_sequences=True)(x)
    
    x = UpSampling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling1D(2)(x)
    outputs = Conv1D(input_shape[-1], 3, activation='linear', padding='same')(x)
    
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder



def classify_with_radius_neighbors(featuresNlabels, radius=1.0):
    encoded_features = []
    labels = []
    # Encode the features
    
    mfcc_features = [mfcc[0] for mfcc in featuresNlabels]
    mfcc_features = np.stack(mfcc_features, axis=0)

    input_shape = mfcc_features[0].shape
    autoencoder, encoder = crnn_encoder(input_shape)

    # Compile the autoencoder
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the autoencoder
    autoencoder.fit(mfcc_features, mfcc_features, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2, 
                    shuffle=True)
    encoded_f = encoder.predict(mfcc_features)
    encoded_features.append(encoder.predict(mfcc_features))
    
    for _, label in featuresNlabels: # featuresNlabels.shape = (n_samples, )
        labels.append(label)
    
    # Convert to numpy arrays
    encoded_features = np.array(encoded_features)[0]
    labels = np.array(labels)

    clf = RadiusNeighborsClassifier(radius=radius)
    clf.fit(encoded_features, labels)

    # Predict the labels using the same features (to check clustering quality)
    predicted_labels = clf.predict(encoded_features)
    
    # Print cluster information
    unique_labels = np.unique(predicted_labels)
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
    plt.savefig('dec_crnn_results.png')
    plt.show()
    
    # Evaluate accuracy of clustering
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return predicted_labels, accuracy



if __name__=='__main__':
    featuresNlabels = load_audio(root_dir='audio')
    predicted_labels, accuracy = classify_with_radius_neighbors(featuresNlabels)