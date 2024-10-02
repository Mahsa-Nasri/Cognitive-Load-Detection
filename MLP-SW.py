
"""
Created on Thu Jul 25 13:32:52 2024

@author: mahsa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the combined CSV file
combined_csv = 'MergedData.csv'

# Step 1: Load the combined CSV file
data = pd.read_csv(combined_csv)

# Define the window size
window_size = 2000   

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
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1, stratify=y)

# Number of features (dimensions)
n_input = data_x_train.shape[1]

# Step 5: Build MLP (feedforward) neural network architecture
x_input = tf.keras.layers.Input(shape=(n_input,))
x_hidden_1 = tf.keras.layers.Dense(units=64, activation='tanh')(x_input)
x_hidden_2 = tf.keras.layers.Dense(units=32, activation='tanh')(x_hidden_1)
x_hidden_3 = tf.keras.layers.Dense(units=16, activation='tanh')(x_hidden_2)
#x_hidden_4 = tf.keras.layers.Dense(units=8, activation='tanh')(x_hidden_3)
#x_hidden_5 = tf.keras.layers.Dense(units=4, activation='tanh')(x_hidden_4)

x_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x_hidden_3)  # Using sigmoid for binary classification
model = tf.keras.Model(inputs=[x_input], outputs=[x_output], name='mlp')

# Print summary of model architecture
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=['accuracy'])

# Step 6: Train model
history = model.fit(data_x_train, data_y_train, epochs=1000, batch_size=256, validation_data=(data_x_test, data_y_test))

# Predict on training and test data
train_predictions = (model.predict(data_x_train) > 0.5).astype("int32")
test_predictions = (model.predict(data_x_test) > 0.5).astype("int32")

# Calculate accuracy, precision, recall, and F1 score for training data
accuracy_train = accuracy_score(data_y_train, train_predictions)
precision_train = precision_score(data_y_train, train_predictions)
recall_train = recall_score(data_y_train, train_predictions)
f1_train = f1_score(data_y_train, train_predictions)

# Calculate accuracy, precision, recall, and F1 score for test data
accuracy_test = accuracy_score(data_y_test, test_predictions)
precision_test = precision_score(data_y_test, test_predictions)
recall_test = recall_score(data_y_test, test_predictions)
f1_test = f1_score(data_y_test, test_predictions)

print("=================================================")
print("MLP Training Metrics: Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}".format(accuracy_train, precision_train, recall_train, f1_train))
print("MLP Test Metrics: Accuracy = {}, Precision = {}, Recall = {}, F1 Score = {}".format(accuracy_test, precision_test, recall_test, f1_test))
print("=================================================")

# Plot confusion matrix for test data
conf_matrix = confusion_matrix(data_y_test, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('mlp_loss.png')
plt.show()
