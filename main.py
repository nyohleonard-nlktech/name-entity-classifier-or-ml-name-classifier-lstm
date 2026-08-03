import numpy as np
import pickle
import json
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

app = FastAPI()
model = load_model('model_attention_final.keras', custom_objects={'AttentionLayer': AttentionLayer})

with open('tokenizer.json', 'r') as f:
    tk_data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tk_data)

with open('label_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

class Query(BaseModel):
    name: str

@app.post("/predict")
def predict(data: Query):
    seq = tokenizer.texts_to_sequences([data.name])
    padded = pad_sequences(seq, maxlen=15, padding='post')
    probs = model.predict(padded, verbose=0)
    idx = np.argmax(probs)
    return {
        "input": data.name,
        "prediction": str(classes[idx]),
        "confidence": f"{float(np.max(probs))*100:.2f}%"
    }
