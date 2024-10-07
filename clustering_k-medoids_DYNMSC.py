import os
import joblib
import kmedoids
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from clustering_preProcess import *



def train_dynmsc(features, min_k=2, max_k=5, max_iter=100, metric='cosine'):
    """
    Train using DynMSC (Dynamic Medoid Silhouette Clustering) from kmedoids with automatic cluster number selection.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute dissimilarity matrix using a specified distance metric (e.g., cosine, euclidean).
    dist_matrix = squareform(pdist(features_scaled, metric))

    # Perform DynMSC clustering
    result = kmedoids.dynmsc(dist_matrix, medoids=max_k, minimum_k=min_k, max_iter=max_iter, init='random')
    
    return result, scaler



def classify_new_person_dynmsc(result, scaler, new_features, all_features, metric='cosine'):
    """
    Classify a new person based on the closest medoid from the DynMSC result.
    """
    # Standardize new features using the same scaler
    new_features_scaled = scaler.transform(new_features)
    
    # Extract the feature vectors of the medoids using their indices from the result
    medoid_feature_vectors = all_features[result.medoids]
    
    # Compute dissimilarity of new features to the medoid feature vectors
    combined_features = np.vstack([medoid_feature_vectors, new_features_scaled])
    dist_matrix = squareform(pdist(combined_features, metric))
    
    # Get the dissimilarity matrix for the new features to medoids (rows 1 onward, columns only for medoids)
    cluster_labels = np.argmin(dist_matrix[len(medoid_feature_vectors):, :len(medoid_feature_vectors)], axis=1)
    
    counts = Counter(cluster_labels)
    majority_vote = counts.most_common(1)[0][0]
    
    return majority_vote if majority_vote != -1 else 'Unknown'



def evaluate_clustering_accuracy(result, scaler, features, labels, metric='cosine'):
    """
    Evaluate clustering accuracy by comparing predicted labels with true labels.
    """
    # Standardize the features using the same scaler
    features_scaled = scaler.transform(features)
    
    # Extract the feature vectors of the medoids using their indices from the result
    medoid_feature_vectors = features[result.medoids]
    
    # Compute dissimilarity of features to medoids
    combined_features = np.vstack([medoid_feature_vectors, features_scaled])
    dist_matrix = squareform(pdist(combined_features, metric))
    
    # Get the dissimilarity matrix for the features to medoids
    predicted_labels = np.argmin(dist_matrix[len(medoid_feature_vectors):, :len(medoid_feature_vectors)], axis=1)
    print(predicted_labels)
    # Calculate and return the clustering accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Clustering Accuracy: {accuracy}")
    
    return accuracy



if __name__ == "__main__":
    base_dir = "audio"
    train_features, test_features, train_labels,test_labels= load_audio_from_dirs(base_dir) #TODO - why do we need it in classify_new_person_dynmsc ? Seems redungant. Put in else only

    classifier = 'dynmsc_classifier.pkl'
    if os.path.exists(classifier):
        result = joblib.load('dynmsc_classifier.pkl')
        scaler = joblib.load('dynmsc_scaler.pkl')
    else:
        result, scaler = train_dynmsc(train_features, min_k=2, max_k=5, max_iter=100, metric='cosine')
        joblib.dump(result, classifier)
        joblib.dump(scaler, 'dynmsc_scaler.pkl')

    test_dir = "test"
    test_features = load_audio_from_test_dir(test_dir) #TODO
    new_person_cluster_labels = classify_new_person_dynmsc(result, scaler, test_features, train_features, metric='cosine')
    accuracy = evaluate_clustering_accuracy(result, scaler, test_features, test_labels, metric='cosine')
    print("New person classified as: ", new_person_cluster_labels)



'Based on https://python-kmedoids.readthedocs.io/en/latest/#dynmsc'
'DynMSC (Dynamic Medoid Silhouette Clustering)'


'https://arxiv.org/pdf/2008.05171'

'''    TODO - new : Look above
    . Clean code: 
    0. sort out all functions in visualize (seperate HDBSCAN)
    1. Have only 1 "load from dirs" i.e. remove load audio
    2. Move all tests functions from all models to visualize (and send to visualize only predicts, and labels for graphs, and model for evaluation)
    3. Add custom names for each graph based on the model. 
    - finish adding requirements
'''

'''
What we did:
1. Extracted features
2. Visualized data (files in first commit)
3. Used clustering and more recently radius_neighbors_classifier to classify new drivers (including unkown)
4. Created an evaluation and prediction function using the trained classifier
5. Clean code, merge w/ "first" commit (to print graphs w/ variance) 
6. Save created figure of the dataset after PCA and clustering (first commit) + understand what it means 
7. Used Deep Embedded Clustering with EfficientNet pre-trained model as an auto-encoder and RadiusNeighborsClassifier
based on 'Related paper: https://ieeexplore.ieee.org/document/9538747'
8. Performed initial visualization
9. Used DynMSC 

'''