Visual Question Answering

This repository contains code for building and training a Visual Question Answering (VQA) model using Keras and TensorFlow. The model takes an image and a question as input and predicts the answer to the question based on the image content.

Requirements:
- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- tqdm

Dataset:
The code is designed to work with a VQA dataset in the following format:
- A CSV file containing columns for 'question', 'answer', and 'image_id'
- A directory containing the corresponding images, with filenames matching the 'image_id' column in the CSV file

Usage:
1. Clone the repository and navigate to the project directory.
2. Place your dataset files (CSV and image directory) in the appropriate location specified in the code.
3. Run the code to preprocess the data, build the model, and train it on your dataset.
4. After training, you can use the predict_answer function to make predictions on new image-question pairs.

Code Overview:
1. The load_data function reads the CSV file and loads the images, preprocessing them using the VGG16 model.
2. The text data (questions) is preprocessed by tokenizing the words, converting them to sequences of integers, and padding the sequences to a fixed length.
3. The answer labels are encoded by creating a mapping dictionary from labels to integers and converting the integer-encoded labels to a one-hot encoded format.
4. The VQA model architecture is defined, consisting of two branches: one for processing the image input (using VGG16) and another for processing the text input (questions) using an Embedding layer, Bidirectional LSTM, and LSTM layers.
5. The model is compiled with the Adam optimizer and categorical cross-entropy loss function, and the accuracy metric is specified for evaluation.
6. The model is trained using the preprocessed training data and saved to a file ('vaq.h5').
7. Helper functions (preprocess_image, preprocess_question, predict_answer) are provided for preprocessing new inputs and making predictions using the trained model.

Note: This code is intended for educational and research purposes only. It may require modifications or additional steps to work with your specific dataset or environment.
