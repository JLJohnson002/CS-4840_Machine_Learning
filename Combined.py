import torch
from PIL import Image
import os
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

os.system('cls')#FIXME

folder_path = r"Cropped Images/Not Death Star"
print_path = r"Identified Images"
itter = 0

for item in os.listdir(folder_path):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Images
    item_path = os.path.join(folder_path, item)
    # img = "Images\DeathStar3.jpeg"  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    model.conf = 0.2
    results = model(item_path, size=320) #old was 320

    # Results
    # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

    # Load the original image
    ori_img = Image.open(item_path)
    for i, (*box, conf, cls) in enumerate(results.xyxy[0]):  # xyxy format
        x1, y1, x2, y2 = map(int, box[:4])  # Extract bounding box
        cropped_img = ori_img.crop((x1, y1, x2, y2))  # Crop region
        cropped_img.convert("RGB").save(f"{item[:-4]}_cropped_{i}.png")  # Save the cropped image
    itter += 1

print ("\n YOLO FINISHED\n")

# Define a function to extract features (for example, using color histograms)
def extract_features(image):
    # Convert the image to RGB (in case it's in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image for consistency
    image = cv2.resize(image, (512,512))  #ADJUST Resizing to 128x128 for simplicity
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
    print (len(images))
    print (len(labels))
    return np.array(images), np.array(labels)


# Path to your image folder (where each subfolder is a class)
# image_folder = "Mixed Images"
# image_folder = "Images"
image_folder = "Cropped Images"

# Load images and labels
X, y = load_images_from_folder(image_folder)
print ("loaded")


# Encode the labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print ("transformed")

# Optionally, apply PCA for dimensionality reduction (if features are large)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X,y)
print ("scaled")

# Use PCA to reduce the dimensionality if necessary (e.g., for large image sizes)
pca = PCA(n_components=1)  #ADJUST You can adjust the number of components
X_pca = pca.fit_transform(X_scaled)
print ("pca")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.1, random_state=42)
print ("split")

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)  #ADJUST You can tune the number of neighbors
print ("init")

# Train the KNN classifier
knn.fit(X_train, y_train)
print ("fit")

# Make predictions
y_pred = knn.predict(X_test)
print ("predicted")

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

# Example: Predict a new image
# for each in os.listdir("Cropped Images\\Death Star"):
#     # print (each)
#     new_image_path = "Cropped Images\\Death Star\\"+str(each)
#     predicted_label = predict_image(new_image_path)
#     print(f"Predicted Label {new_image_path}: {predicted_label}")
