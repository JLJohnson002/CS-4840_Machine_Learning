# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

# Step 1: Generate a synthetic dataset with 2 features and 4 centers (clusters)
X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=4)

# Step 2: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the KNN classifier with 5 neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# Step 4: Train the KNN model using the training data
knn.fit(X_train, y_train)

# Step 5: Visualize the decision regions of the trained KNN model
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=knn, legend=2)

# Step 6: Label the axes and add a title to the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN with K=5')

# Step 7: Save the plot as an image file with tight bounding box and high resolution (150 dpi)
plt.savefig('KNN with K=5.jpeg', bbox_inches="tight", dpi=150)

# Step 8: Display the plot
plt.show()