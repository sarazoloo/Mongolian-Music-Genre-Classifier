
#SizeRestrictions_BODY.
import streamlit as st
import isodate

import pickle
import pytube as pt
from pytube import YouTube
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from tempfile import TemporaryFile
import random 
import operator
import math

import librosa
import librosa.display

import json


sns.set(style="darkgrid", color_codes=True)
st.set_page_config(layout="wide")

st.title('Mongolian Music Genre Classifier')


st.header("Description")
st.markdown(f""" ### This is a simple music classifier with around 6000 audio files and 6 different genres.
Each audio file is 30 seconds long. The 6 genres are:
\n Pop, mpop, rock, indie, folk, and hiphop.
Although there are a lot of genres mongolian most songs fall into the genres pop, and hiphop.
If you are wondering about what mpop is it's mongolian pop, slightly different from 
the international pop music your used to.
""")
st.markdown("##### The model is based on the mfcc put in 10 different segments. Here is sample of how each genre's MFCCs look like")
image_folk = Image.open('/folk_MFCC.png')
image_hiphop = Image.open('/hiphop_MFCC.png')
image_pop = Image.open('/pop_MFCC.png')
image_mpop = Image.open('/mpop_MFCC.png')
image_rock = Image.open('/rock_MFCC.png')
image_indie = Image.open('/indie_MFCC.png')
st.image(image_folk, caption ="Folk genre", use_column_width=True)
st.image(image_pop, caption ="Pop genre", use_column_width=True)
st.image(image_mpop, caption ="Mpop genre", use_column_width=True)
st.image(image_rock, caption ="Rock genre", use_column_width=True)
st.image(image_indie, caption ="Indie genre", use_column_width=True)

#load model
ann_model = pickle.load(open('ann_model.pkl', 'rb'))

#genre_dict for the output
genre_dict = {0:'hiphop', 1:'rock', 2:'mpop', 3:'folk', 4:'pop', 5:'indie'}

#url to mp3 converter
def youtube_to_mp3(url):    
    yt = YouTube(str(url))
    video = yt.streams.filter(only_audio=True).first()
    #print("Enter the destination address (leave blank to save in current directory)")
    #destination = str(input(" ")) or '.'
    out_file = video.download(output_path='.')
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    
    return print(yt.title + " has been successfully loaded ")

st.header("File upload")
uploaded_file = st.file_uploader("Please upload a .mp3")
st.write (f""" Upload a file or upload a URL""")

try:
    if uploaded_file is not None:
        audio_file = uploaded_file
        st.audio( audio_file)
    else:
        yt_url = st.text_input("Input youtube url")
        if len(yt_url) > 0:
            st.audio(yt_url)
            audio_file = youtube_to_mp3(yt_url)
except:
    # print("File upload error")
    raise

#processing audio file to get mfcc
def process_input(audio_file):
    sample_rate = 22050
    num_mfcc = 13
    n_ftt=2048
    hop_len=512
    track_dur = 30 # measured in seconds
    samples_per_track = sample_rate * track_dur
    num_seg = 10
    samples_per_segment = int(samples_per_track / num_seg)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_len)

    signal, sample_rate = librosa.load(audio_file, sr=sample_rate)
    for d in range(10):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

            # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_ftt, hop_length=hop_len)
        mfcc = mfcc.T

    return mfcc

    #prediction func
def predict(X_predict):
    X_to_predict = X_predict[np.newaxis, ...]

    predict_prob=ann_model.predict(X_to_predict)
    predict_classes=np.argmax(predict_prob, axis=1)
    # get index with max value
    predicted_index = (predict_classes)

    pred_percen = round((predict_prob.max() * 100), 2)

    #print("Predicted Genre:", genre_dict[int(predicted_index)] , predict_prob.max())
    pred_genre = genre_dict[int(predicted_index)]
    pred_prob = predict_prob.max()
    return pred_genre, pred_percen

        
predict_button = st.button("Predict music genre", key = '01')
#prediction button
if predict_button:
    input_mfcc = process_input(audio_file)
    genre, percentage = predict(X_predict=input_mfcc)

    st.write("Prediction results", genre, percentage, "%")

else:
    st.write('No files uploaded')
