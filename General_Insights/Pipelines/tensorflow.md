

# Example of using `tensorflow`'s pipeline to take care of leakage.

```Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example: generate synthetic data
X, y = np.random.rand(1000, 10), np.random.randint(0, 2, 1000)

# Step 1: Split FIRST (avoid leakage!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit scalers on training data ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use same scaler!

# Step 3: Build a simple model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model on training data
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Step 5: Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print("Test accuracy:", test_acc)
```

