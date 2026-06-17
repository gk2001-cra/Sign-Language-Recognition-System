from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import os

# Create label mapping
label_map = {label: num for num, label in enumerate(actions)}

X = []
y = []

# Load all .npy files
for action in actions:

    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        print(f"Folder not found: {action_path}")
        continue

    for file in os.listdir(action_path):

        if file.endswith(".npy"):

            data = np.load(
                os.path.join(action_path, file)
            )

            X.append(data)
            y.append(label_map[action])

# Convert to numpy arrays
X = np.array(X)

y = to_categorical(y).astype(int)

print("X Shape =", X.shape)
print("Y Shape =", y.shape)

print("Classes =", actions)
print("Number of Classes =", actions.shape[0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Model
model = Sequential([
    Input(shape=(63,)),

    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),

    Dense(actions.shape[0], activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Save Model
model.save("model.h5")

print("\nModel Saved Successfully!")
print("File Created: model.h5")