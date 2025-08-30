import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Parameters (adjust if needed)
vocab_size = 5000
embedding_dim = 64
input_length = 20
num_classes = 10  # Placeholder, will adjust later if needed

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Save the model
model.save('models/base_model.h5')

print("✅ Base model created and saved to 'models/base_model.h5'")
