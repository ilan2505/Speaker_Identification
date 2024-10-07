import os
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsClassifier
from clustering_preProcess import *



def train_radius_neighbors_classifier(features, labels, radius=1.0):
    """
    Train RadiusNeighborsClassifier with integer outlier labels.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Radius Neighbors Classifier with integer outlier_label=-1
    radius_classifier = RadiusNeighborsClassifier(radius=radius, outlier_label=-1)
    radius_classifier.fit(features_scaled, labels)

    return radius_classifier, scaler



def classify_new_person(radius_classifier, scaler, new_features):
    # Standardize new features using the same scaler
    new_features_scaled = scaler.transform(new_features)

    # Predict cluster labels for new features using the RadiusNeighborsClassifier
    cluster_labels = radius_classifier.predict(new_features_scaled)
    
    counts = Counter(cluster_labels)

    majority_vote = counts.most_common(1)[0][0]

    return majority_vote if majority_vote != -1 else 'Unknown'



if __name__ == "__main__":

    classifier = 'radius_classifier.pkl'
    if os.path.exists(classifier):
        radius_classifier = joblib.load('radius_classifier.pkl')
        scaler = joblib.load('radius_scaler.pkl')  
    else:
        base_dir = "audio"  
        train_features, test_features, train_labels,test_labels = load_audio_from_dirs(base_dir)
        radius_classifier, scaler = train_radius_neighbors_classifier(train_features, train_labels, radius=1.0)  # Adjust the radius
        joblib.dump(radius_classifier, classifier)
        joblib.dump(scaler, 'radius_scaler.pkl')

    test_dir = "1-test" 
    test_features = load_audio_from_test_dir(test_dir)
    for person in test_features:#TODO
        new_person_cluster_labels = classify_new_person(radius_classifier, scaler, test_features)
    print("New person classified as: ", new_person_cluster_labels)