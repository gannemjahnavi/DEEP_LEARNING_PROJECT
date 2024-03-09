import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SimpleRNN
from tensorflow.keras.datasets import mnist
import os

# Load MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

# Reshape data to have a single channel
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Normalize pixel values
X_train = X_train / 255.0

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define model architectures
models = [
    {
        'name': 'CNN',
        'model': Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    },
    {
        'name': 'ANN',
        'model': Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    },
    {
        'name': 'RNN',
        'model': Sequential([
            SimpleRNN(128, input_shape=(28, 28), activation='relu'),
            Dense(10, activation='softmax')
        ])
    }
]

# Train models
for model_info in models:
    model_name = model_info['name']
    model = model_info['model']
    print(f"Training {model_name} model...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    save_model(model, os.path.join(script_dir, f'{model_name.lower()}_model'))

print("All models trained successfully.")
