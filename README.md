# Name Entity Classifier — LSTM with Attention

A Streamlit demonstration of a trained Keras sequence-classification model that predicts a category for a user-entered name.

## Supported labels
The interface describes the model classes as Male, Female, Country, Animal, or Other. The actual label order and class definitions are loaded from label_classes.pkl.

## Model pipeline
1. Load the saved Keras model from model_attention_final.keras.
2. Reconstruct the tokenizer from tokenizer.json.
3. Load the class labels from label_classes.pkl.
4. Convert the input string to a token sequence.
5. Pad the sequence to 15 tokens.
6. Run model inference and display the highest-scoring class and model confidence.

The repository defines a custom AttentionLayer needed to load the saved model.

## Technology
Python, TensorFlow/Keras, NumPy, Streamlit, and pickle/JSON model artifacts.

## Requirements
Use a Python version supported by the TensorFlow release installed in your environment. TensorFlow may require platform-specific setup.

## Setup
```bash
git clone https://github.com/nyohleonard-nlktech/name-entity-classifier-or-ml-name-classifier-lstm.git
cd name-entity-classifier-or-ml-name-classifier-lstm
python -m venv .venv
```

Activate:
```bash
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install:
```bash
pip install -r requirements.txt
```

## Run the Streamlit interface
```bash
streamlit run streamlit_app.py
```
The repository also contains app.py with the same Streamlit inference workflow.

## Required artifacts
```text
model_attention_final.keras
tokenizer.json
label_classes.pkl
```
Keep the model, tokenizer, and class-label artifacts from the same training run. Their vocabularies, sequence length, and class order must remain aligned.

## Limitations and responsible use
This is a name-pattern classifier, not a reliable way to determine a person's gender, identity, nationality, or other sensitive attributes. Names are culturally diverse and ambiguous; predictions may be wrong and should not be used for decisions about people. Confidence is the model's output score, not a calibrated probability.

No license file is present. All rights reserved unless an explicit license is added.
