import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import datasets, neighbors


# Function to extract features
def extract_features(image):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image for consistency
    image = cv2.resize(image, (32, 32))  # ADJUST
    # Flatten the image to a 1D vector 
    return image.flatten()

# Function to load images and labels from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(extract_features(image))
                    labels.append(label)
    return np.array(images), np.array(labels)

# Function to predict a new image
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        features = extract_features(image)
        features_scaled = scaler.transform([features])  # Scale the features
        features_pca = pca.transform(features_scaled)  # Apply PCA
        prediction = knn.predict(features_pca)
        return le.inverse_transform(prediction)[0]  # Return the predicted class label

os.system("cls")

cycles = [1]
for each in cycles:
    # print ("Started")
    image_folder = os.path.join(os.path.expanduser("~"), "Desktop", "cifar10_images")

    # Load images and labels
    X, y = load_images_from_folder(image_folder)
    # print("loaded")

    # Encode the labels into integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # print("transformed")

    # Optionally, apply PCA for dimensionality reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X, y)
    # print("scaled")

    # Use PCA to reduce the dimensionality
    components = 40
    pca = PCA(n_components=components)  # ADJUST 
    X_pca = pca.fit_transform(X_scaled)
    # print("pca")

    # Split the dataset into training and testing sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=test_size, random_state=42 # ADJUST
    )
    # print("split")

    # Initialize KNN classifier
    k_neighbors = 11
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)  # ADJUST
    # print("init")

    # Train the KNN classifier
    knn.fit(X_train, y_train)
    # print("fit")

    # Make predictions
    y_pred = knn.predict(X_test)
    # print("predicted")
    # Print classification report to evaluate performance
    print ("Components: "+ str(components))
    print ("Test size: "+ str(test_size))
    print ("Number of neighbors(K): "+ str(k_neighbors))
    print(classification_report(y_test, y_pred, target_names=le.classes_))