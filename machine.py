# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print("Evaluation Results:")
print("-" * 40)
print(f"Accuracy: {accuracy:.2%}\n")

# Print the Classification Report with better formatting
print("Classification Report:")
print("-" * 60)
print("{:<15} {:<15} {:<15} {:<15} {:<15}".format("Class", "Precision", "Recall", "F1-Score", "Support"))
print("-" * 60)
for i in range(len(iris.target_names)):
    print("{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15}".format(
        iris.target_names[i], 
        precision_score(y_test == i, y_pred == i),
        recall_score(y_test == i, y_pred == i),
        f1_score(y_test == i, y_pred == i),
        np.sum(y_test == i)
    ))

print("-" * 60)
