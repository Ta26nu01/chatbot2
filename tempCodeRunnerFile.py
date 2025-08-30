import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scripts.utils import load_and_clean_data

# Load local Q&A
questions, answers = load_and_clean_data("data/information.txt")

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X = pad_sequences(tokenizer.texts_to_sequences(questions), padding='post')

# Label encoding
le = LabelEncoder()
y = le.fit_transform(answers)

# Load base model
model = tf.keras.models.load_model("models/base_model.h5")

# Compile & Train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=10)

# Save updated model
model.save("models/updated_model_1.h5")  # replace with 2,3,4 on other devices
