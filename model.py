# model.py
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(100,)),  # input + hidden
        Dense(10, activation='softmax')  # output
    ])
    return model
