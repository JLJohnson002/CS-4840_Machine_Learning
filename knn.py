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


# Define a function to extract features (for example, using color histograms)
def extract_features(image):
    # Convert the image to RGB (in case it's in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image for consistency
    image = cv2.resize(image, (512, 512))  # ADJUST Resizing to 128x128 for simplicity
    # Flatten the image to a 1D vector (128x128x3 pixels)
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
    print(len(images))
    print(len(labels))
    return np.array(images), np.array(labels)


os.system("cls")

# Path to your image folder (where each subfolder is a class)
# image_folder = "Mixed Images"
# image_folder = "Images"

image_folder = "CroppedTrainImages"
# image_folder = "OriginlTrainImages"

# Load images and labels
X, y = load_images_from_folder(image_folder)
print("loaded")


# Encode the labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("transformed")

# Optionally, apply PCA for dimensionality reduction (if features are large)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X, y)
print("scaled")

# Use PCA to reduce the dimensionality if necessary (e.g., for large image sizes)
components = 12 #ADJUST
pca = PCA(n_components=components)  # ADJUST You can adjust the number of components
X_pca = pca.fit_transform(X_scaled)
print("pca")

# Split the dataset into training and testing sets
test_size_decimal = 0.1 #ADJUST
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=test_size_decimal, random_state=42
)
print("split")

# Initialize KNN classifier
number_of_neighbors = 2
knn = KNeighborsClassifier(n_neighbors=number_of_neighbors)  # ADJUST You can tune the number of neighbors
print("init")

# Train the KNN classifier
knn.fit(X_train, y_train)
print("fit")

# Make predictions
y_pred = knn.predict(X_test)
print("predicted")

# Print classification report to evaluate performance
print("report start")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("report end")


# To predict a new image, use the following:
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        features = extract_features(image)
        features_scaled = scaler.transform([features])  # Scale the features
        features_pca = pca.transform(features_scaled)  # Apply PCA
        prediction = knn.predict(features_pca)
        return le.inverse_transform(prediction)[0]  # Return the predicted class label


# from mlxtend.plotting import plot_decision_regions

# selected_features = [0, 1]  # Choosing the first two PCA components for visualization

# # Generate filler values (mean of each remaining feature)
# filler_values = np.mean(X_pca, axis=0)  # Mean of each PCA feature
# filler_feature_values = {i: filler_values[i] for i in range(2, 12)}

# filler_ranges = {
#     i: (X_pca[:, i].min(), X_pca[:, i].max()) for i in range(2, 12)
# }  # Range for other features
# filler_feature_ranges = {i: (np.min(X_pca[:, i]), np.max(X_pca[:, i])) for i in range(2, 12)}

# # 6. Plot decision regions
# plt.figure(figsize=(8, 6))

# plot_decision_regions(
#     X_train,
#     y_train,
#     clf=knn,
#     legend=2,
#     X_highlight=None,  # Optional: highlight test samples
#     feature_index=selected_features,  # Select 2 PCA components
#     filler_feature_values=filler_values,  # Mean values for non-plotted dimensions
#     filler_feature_ranges=filler_ranges,  # Min-max range for non-plotted dimensions
# )
# # Step 6: Label the axes and add a title to the plot
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("KNN with K=5")

# # Step 7: Save the plot as an image file with tight bounding box and high resolution (150 dpi)
# plt.savefig("KNN with K=5.jpeg", bbox_inches="tight", dpi=150)

# # Step 8: Display the plot
# plt.show()
running = True

while running:
    # folder_path = input("What is the folder path? ")
    folder_path = "CroppedTrainImages\\Death Star\\"
    wrong_count = 0
    # Use PCA to reduce the dimensionality if necessary (e.g., for large image sizes)
    
    components = int(input ("Compontents 12 high: ")) #ADJUST 12 high 19 max
    test_size_decimal = float(input ("Test Size: ")) #ADJUST
    number_of_neighbors = int(input("Neighbors: "))
    if str(components) == "n":
        running = False
    
    pca = PCA(n_components=components)  # ADJUST You can adjust the number of components
    X_pca = pca.fit_transform(X_scaled)
    print("pca")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=test_size_decimal, random_state=42
    )
    print("split")

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=number_of_neighbors)  # ADJUST You can tune the number of neighbors
    print("init")

    # Train the KNN classifier
    knn.fit(X_train, y_train)
    print("fit")

    # Make predictions
    y_pred = knn.predict(X_test)
    print("predicted")

    # Print classification report to evaluate performance
    print("report start")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("report end")
    print("The following are incorrect")
    print ()
    for each in os.listdir(folder_path):
        new_image_path = folder_path + str(each)
        predicted_label = predict_image(new_image_path)

        if predicted_label != folder_path[19:-1:]:
            wrong_count +=1
            print(new_image_path)

        # print(f"Predicted Label {new_image_path}: {predicted_label}")
    print ()


    print ("Wrong count is "+str(wrong_count))
