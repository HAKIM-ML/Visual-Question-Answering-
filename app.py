import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Function to load model, tokenizer, and unique words
@st.cache(allow_output_mutation=True)
def load_model_tokenizer():
    # Load model
    model = load_model('vaq.h5')
    tokenizer = joblib.load('tokenizer.joblib')
    unique_answers = joblib.load('unique_answers.joblib')
    preprocess_input = joblib.load('preprocesse_input.joblib')
    return model, tokenizer, unique_answers, preprocess_input

model, tokenizer, unique_answers, preprocess_input = load_model_tokenizer()

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.array([img_array]))
    return img_array

# Function to preprocess the question
def preprocess_question(question, tokenizer, max_len=24):
    seq = tokenizer.texts_to_sequences([question])
    seq = pad_sequences(seq, maxlen=max_len, truncating='post', padding='post')
    return seq

# Function to predict the answer
def predict_answer(image, question, model, tokenizer, unique_answers):
    preprocessed_image = preprocess_image(image)
    preprocessed_question = preprocess_question(question, tokenizer)
    prediction = model.predict([preprocessed_image, preprocessed_question])
    predicted_class_idx = np.argmax(prediction)
    predicted_answer = unique_answers[predicted_class_idx]
    return predicted_answer

# Main function to run the Streamlit app
def main():
    # Load model, tokenizer, and unique words
    model, tokenizer, unique_answers, preprocess_input = load_model_tokenizer()
    st.title("Image Question Answering")
    st.write("Welcome to the Image Question Answering App!")

    # Drag and drop image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Get the question from the user
        question = st.text_input("Enter the question:")

        if st.button("Predict Answer"):
            if question:
                predicted_answer = predict_answer(uploaded_file, question, model, tokenizer, unique_answers)
                st.write(f"Predicted answer: {predicted_answer}")
            else:
                st.warning("Please enter a question.")

    if st.button("Clear"):
        st.file_uploader("Choose an image...", type=["jpg", "png"])
        st.text_input("Enter the question:", value="")

if __name__ == "__main__":
    main()