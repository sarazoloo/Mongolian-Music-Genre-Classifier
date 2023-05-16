
#SizeRestrictions_BODY.
import streamlit as st
import isodate
from io import BytesIO
from pathlib import Path

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

st.title('Mongolian Music Genre Classifier :flag-mn:')

st.markdown(f""" ### This is a simple music classifier with around 6000 audio files and 6 different genres.
Each audio file is 30 seconds long. The 6 genres are:
\n Pop, mpop, rock, indie, folk, and hiphop.
Although there are a lot of genres mongolian most songs fall into the genres pop, and hiphop.
If you are wondering about what mpop is it's mongolian pop, slightly different from 
the international pop music your used to. :musical_note:
""")
st.markdown("##### The model is based on the mfcc put in 10 different segments. Here is sample of how each genre's MFCCs look like")
mffc_images = [
    'genreclass/folk_MFCC.png',
    'genreclass/hiphop_MFCC.png',
    'genreclass/pop_MFCC.png',
    'genreclass/mpop_MFCC.png',
    'genreclass/rock_MFCC.png',
    'genreclass/indie_MFCC.png']

image_captions = [
    'Folk MFCC',
    'Hiphop MFCC',
    'Pop MFCC', 
    'Mpop MFCC',
    'Rock MFCC',
    'Indie MFCC']
#image_iterator = paginator("Select a sunset page", mffc_images)
#indices_on_page, images_on_page = map(list, zip(*image_iterator))
st.image(mffc_images, width=600, caption=image_captions)

#load model
ann_model = pickle.load(open('genreclass/ann_model.pkl', 'rb'))

#genre_dict for the output
genre_dict = {0:'hiphop', 1:'rock', 2:'mpop', 3:'folk', 4:'pop', 5:'indie'}

#url to mp3 converter
def youtube_to_mp3(url): 
    try:
        yt = YouTube(str(url))
        video = yt.streams.filter(only_audio=True).first()
        #print("Enter the destination address (leave blank to save in current directory)")
        #destination = str(input(" ")) or '.'
        out_file = video.download(output_path='/genreclass/download')
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)
    except:
        raise
    return print(yt.title + " has been successfully loaded ")



#st.set_page_config(page_title="Download Audio", page_icon="🎵", layout="centered", initial_sidebar_state="collapsed")

@st.cache_data(show_spinner=False)
def download_audio_to_buffer(url):
    buffer = BytesIO()
    youtube_video = YouTube(url)
    audio = youtube_video.streams.get_audio_only()
    default_filename = audio.default_filename
    audio.stream_to_buffer(buffer)
    return default_filename, buffer

def main():
    st.title("Download Audio from Youtube")
    url = st.text_input("Insert Youtube URL:")
    if url:
        with st.spinner("Downloading Audio Stream from Youtube..."):
            default_filename, buffer = download_audio_to_buffer(url)
        st.subheader("Title")
        st.write(default_filename)
        title_vid = Path(default_filename).with_suffix(".mp3").name
        st.subheader("Listen to Audio")
        st.audio(buffer, format='audio/mpeg')
        st.subheader("Download Audio File")
        st.download_button(
            label="Download mp3",
            data=buffer,
            file_name=title_vid,
            mime="audio/mpeg")


st.subheader("Youtube url to mp3 converter")

#try:
#    if url:
#        yt = YouTube(str(url))
#        video = yt.streams.filter(only_audio=True).first()
#        st.download('Download music to your computer', data = video, file_name = f"""{}.mp3""", key = '02')
#    else:
#        if len(yt_url) > 0:
#            st.write("No url uploaded")             
#            audio_file = youtube_to_mp3(yt_url)
#            st.audio(audio_file)
#except:
#    # print("File upload error")
#    raise
                               
st.subheader("File upload")
uploaded_file = st.file_uploader("Please upload a .mp3")
st.write (f""" Upload a file or upload a URL""") 
if uploaded_file is not None:
    audio_file = uploaded_file
    st.audio(audio_file)
else:
    st.write("No file is uploaded")
     

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

    st.write(f"""Prediction results: {genre}
    \n with percentage of: {percentage} "%""")

else:
    st.write('No files uploaded')
    
    
if __name__ == "__main__":
    main()
