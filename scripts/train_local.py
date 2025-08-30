import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv('information.csv')
X = df['prompt'].astype(str).values
y = df['response'].astype(str).values

# Vectorize input prompts
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X).toarray()

# Save vectorizer for later use
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Save label encoder
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Build model
model = Sequential()
model.add(Input(shape=(X_vectorized.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_vectorized, y_categorical, epochs=10, batch_size=8)

# Save model
model.save('models/local_model.h5')
print("✅ Model trained and saved to 'models/local_model.h5'")
