# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:21:33 2024

@author: mahsa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the combined CSV file
combined_csv = 'MergedData.csv'

# Step 1: Load the combined CSV file
data = pd.read_csv(combined_csv)

# Define the window size
window_size = 2000  # Adjust the window size as needed

# Create lists to hold the windowed data
X = []
y = []

# Step 2: Apply sliding windows
for i in range(len(data) - window_size):
    # Append the window of features
    X.append(data.iloc[i:i + window_size, :-1].values.flatten())  # Flatten to create a single feature vector
    # Append the corresponding target value
    y.append(data.iloc[i + window_size, -1])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Print the shapes of the resulting datasets
print("Input features shape (X):", X.shape)
print("Target labels shape (y):", y.shape)

# Step 3: Scale the input features
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_x.fit_transform(X)

# Step 4: Split data into train and test parts
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Step 6: Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt','log2'],
    'bootstrap': [True, False]
}

# Step 7: Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(data_x_train, data_y_train)

# Step 8: Best parameters and model
print("Best parameters found: ", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

# Step 9: Predict on training and test data with the best model
train_predictions = best_rf_model.predict(data_x_train)
test_predictions = best_rf_model.predict(data_x_test)

# Step 10: Calculate accuracy, precision, recall, and F1 score for training data
accuracy_train = accuracy_score(data_y_train, train_predictions)
precision_train = precision_score(data_y_train, train_predictions)
recall_train = recall_score(data_y_train, train_predictions)
f1_train = f1_score(data_y_train, train_predictions)

# Step 11: Calculate accuracy, precision, recall, and F1 score for test data
accuracy_test = accuracy_score(data_y_test, test_predictions)
precision_test = precision_score(data_y_test, test_predictions)
recall_test = recall_score(data_y_test, test_predictions)
f1_test = f1_score(data_y_test, test_predictions)

print("=================================================")
print("RF Training Metrics: Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}".format(accuracy_train, precision_train, recall_train, f1_train))
print("RF Test Metrics: Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}".format(accuracy_test, precision_test, recall_test, f1_test))
print("=================================================")
