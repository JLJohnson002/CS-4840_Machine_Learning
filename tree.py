import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# Path to the dataset folder
dataset_path = "Cropped Images"

# Set image dimensions
image_size = (64, 64)  # Resize to 64x64 pixels

# Initialize lists for data (features) and labels
X = []  # Features (flattened image data)
y = []  # Labels (0 for NOT Death Star, 1 for Death Star)

# Load images from the "Death Star" folder (label = 1)
death_star_folder = os.path.join(dataset_path, "Death Star")
for filename in os.listdir(death_star_folder):
    img_path = os.path.join(death_star_folder, filename)
    img = Image.open(img_path)
    img = img.resize(image_size)  # Resize the image
    img = np.array(img)  # Convert image to numpy array
    X.append(img.flatten())  # Flatten and append to features list
    y.append(1)  # Label "Death Star" as 1

# Load images from the "NOT Death Star" folder (label = 0)
not_death_star_folder = os.path.join(dataset_path, "NOT Death Star")
for filename in os.listdir(not_death_star_folder):
    img_path = os.path.join(not_death_star_folder, filename)
    img = Image.open(img_path)
    img = img.resize(image_size)  # Resize the image
    img = np.array(img)  # Convert image to numpy array
    X.append(img.flatten())  # Flatten and append to features list
    y.append(0)  # Label "NOT Death Star" as 0

# Convert lists to numpy arrays for training
X = np.array(X)
y = np.array(y)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the decision tree classifier with max_depth = 7
clf = DecisionTreeClassifier(max_depth=7, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Display the test images and its predicted label

# for each in range(len(X_test)):

#     plt.imshow(
#         X_test[each].reshape(image_size[0], image_size[1], 3)
#     )  # Reshape to 64x64x3 for display
#     plt.title(
#         f"Predicted Label: {'Death Star' if y_pred[0] == 1 else 'NOT Death Star'}"
#     )
#     plt.show()

loop = 0
for each in range(len(X_test)):
    if y_pred[loop] == 1:
        plt.imshow(
            X_test[each].reshape(image_size[0], image_size[1], 3)
        )  # Reshape to 64x64x3 for display

        plt.title(
            f"Predicted Label: {'Death Star' if y_pred[loop] == 1 else 'NOT Death Star'}"
        )
        plt.show()
    loop += 1
