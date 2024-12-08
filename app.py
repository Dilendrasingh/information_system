import streamlit as st
import numpy as np
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json


# Load tokenizer and model
@st.cache_resource
def load_resources():
    # Load tokenizer
    with open("tokenizer1.json", "r") as f:
        tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

    # Load the model
    model = load_model("model1.h5")
    return tokenizer, model


# Predict next words
def generate_sequence(text, tokenizer, model, max_length=63):
    for _ in range(10):  # Generate 10 words
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_length, padding="pre")
        pos = np.argmax(model.predict(padded_token_text))

        # Find the word corresponding to the predicted index
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                break
    return text


# Load tokenizer and model
tokenizer, model = load_resources()

# Streamlit interface
st.title("Text Prediction App")
st.write("Enter a seed text, and the model will predict the next words.")

# Input text
input_text = st.text_input("Seed Text", "Overall, an effective information system")

# Generate predictions
if st.button(" Boom !!!Generate Text"):
    with st.spinner("Generating..."):
        result = generate_sequence(input_text, tokenizer, model)
    st.success("Prediction Complete!")
    st.write("**Predicted Text:**")
    st.write(result)
