#Step:1 
# Import all libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.utils import pad_sequences

#Load imdb datset word index

word_index=imdb.get_word_index()
reverse_index={value:key for key,value in word_index.items() }

#Load pretrained model

model=load_model('simple_rnn.h5')

#Step:2 
#Helper function

#Function to decode reviews

def decode_reviews(encoded_review):
    return ' '.join([reverse_index.get(i-3,'?')for i in encoded_review])

#Function for preprocess user output

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#Prediction function

def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    
    sentiment='Positive' if prediction[0][0]>0.5 else 'Nagative'
    
    
    return sentiment, prediction[0][0]

#Streamlit app

import streamlit as st
st.title(" IMDB Movie review analysis")
user_input=st.text_area("Movie review")
if st.button("Classify"):
    preprocess_input=preprocess_text(user_input)
    
        #Make prediction
    pred=model.predict(preprocess_input)
    sentiment='Positive' if pred[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {pred[0][0]}')
    
