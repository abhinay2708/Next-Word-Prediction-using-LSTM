import streamlit as st
from tensorflow.keras.models import load_model
# from tensorflow.keras.initializers import Orthogonal 
import tensorflow as tf
import numpy as np
import pickle

# Load model and tokenizer
model = load_model('nwp.h5', compile=False)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
vocab_array = np.array(list(tokenizer.word_index.keys()))

def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.argmax(model.predict(text_padded), axis=-1)[0]
        text += " " + vocab_array[prediction - 1]
    return text

st.title("Next Word Prediction")
input_text = st.text_input("Enter seed text:")
n_words = st.number_input("Number of words to predict:", min_value=1, max_value=50, value=10)
if st.button("Predict"):
    result = make_prediction(input_text, n_words)
    st.write(result)

