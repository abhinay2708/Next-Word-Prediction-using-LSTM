# Next Word Prediction (NWP) using LSTM

## Project Overview
This project implements a **Next Word Prediction (NWP)** model using deep learning techniques. The model predicts the most likely next word in a given sequence of text. It is trained on a text corpus and uses **LSTM (Long Short-Term Memory)** layers to capture sequential dependencies and contextual relationships between words. The project can be used for text generation, autocomplete features, and conversational AI systems.

---

## Dataset
- The dataset consists of textual data collected for training the model.
- Text preprocessing steps include:
  - Converting all text to lowercase.
  - Removing unnecessary characters or punctuation.
  - Tokenizing text into sequences of words.
- Each sequence is split into **input (X)** and **output (y)** pairs, where:
  - `X` = a sequence of words.
  - `y` = the next word in the sequence.
- Sequences with only one word are dropped.
- Input sequences are **padded** to a fixed length and output labels are **one-hot encoded**.

---

## Features
- Uses a **Tokenizer** to convert words into integer sequences.
- **Embedding layer** maps words to dense vectors.
- **Stacked LSTM layers** learn temporal dependencies.
- Dense layers with ReLU and Softmax activations for prediction.
- Can predict multiple words iteratively for text generation.

---

## Model Architecture
- **Embedding Layer**: Converts words into 14-dimensional vectors.
- **LSTM Layer 1**: 100 units, returns sequences.
- **LSTM Layer 2**: 100 units, returns the final state.
- **Dense Layer**: 100 neurons with ReLU activation.
- **Output Layer**: Softmax activation over the vocabulary.

---

## Installation & Setup
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2. Install required packages:
    ```bash
    pip install tensorflow numpy
    ```
3. (Optional) Use **Google Colab** with GPU for faster training.

---

## Usage
1. **Training the Model**  
   ```python
   model.fit(X, y, epochs=250)
   model.save('nwp.h5')
