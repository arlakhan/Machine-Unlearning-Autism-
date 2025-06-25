| Column Name     | Data Type | Description                                                                                            |
| --------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| `Age_Months`    | Integer   | Age of the newborn baby in months (range: 0 to 48).                                                    |
| `Speaking`      | Float     | Score representing the baby's speaking ability (range: 0 to 100).                                      |
| `Listening`     | Float     | Score representing the baby's listening ability (range: 0 to 100).                                     |
| `Writing`       | Float     | Score representing the baby's early writing/motor-symbol ability (0–100).                              |
| `Hand_Movement` | Float     | Score representing hand movement coordination (range: 0 to 100).                                       |
| `Autism_Risk`   | Integer   | **(Binary Target)**: 1 = High Risk of Autism, 0 = Low Risk (based on low speaking & listening scores). |



# autism_unlearning_project.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("autism_development_dataset.csv")

# Label: Simulate autism risk (1 = High Risk, 0 = Low Risk)
data["Autism_Risk"] = ((data["Speaking"] < 30) & (data["Listening"] < 30)).astype(int)

# Features and target
X = data[["Age_Months", "Speaking", "Listening", "Writing", "Hand_Movement"]].values
y = data["Autism_Risk"].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # CNN expects 3D input

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot learning curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------
# Simulate Machine Unlearning
# ------------------------------------------

# Step: Forget samples with autism risk = 1
forget_indices = data[data['Autism_Risk'] == 1].index
data_unlearned = data.drop(index=forget_indices)

# Re-train on unlearned data
X_u = data_unlearned[["Age_Months", "Speaking", "Listening", "Writing", "Hand_Movement"]].values
y_u = data_unlearned["Autism_Risk"].values

X_u_scaled = scaler.fit_transform(X_u)
X_u_scaled = X_u_scaled.reshape((X_u_scaled.shape[0], X_u_scaled.shape[1], 1))

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_u_scaled, y_u, test_size=0.2, random_state=42)

# New model after unlearning
model_unlearn = tf.keras.models.clone_model(model)
model_unlearn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_u = model_unlearn.fit(X_train_u, y_train_u, validation_data=(X_test_u, y_test_u), epochs=10, batch_size=64)

# Evaluate
loss_u, accuracy_u = model_unlearn.evaluate(X_test_u, y_test_u)
print(f"Post-Unlearning Accuracy: {accuracy_u:.4f}")
