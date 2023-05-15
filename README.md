# Mongolian-Music-Genre-Classifier

#### This is a simple music classifier with around 6000 audio files and 6 different genres.

Each audio file is 30 seconds long. The 6 genres are:

Pop, mpop, rock, indie, folk, and hiphop. Although there are a lot of genres mongolian most songs fall into the genres pop, and hiphop. If you are wondering about what mpop is it's mongolian pop, slightly different from the international pop music your used to.

The model is based on the mfcc put in 10 different segments. Here is sample of how each genre's MFCCs look like:

#### Why we use MFCC:

MFCCs divide the audio signal into short time frames and applying the Fourier transform to each frame. Its also becomes non-sensitive to small changes and variations in the spectrum. MFCCs extract relevant features from the audio that are differentiate the different music genres. This is why we want the MFCC feature.

### Data Preparation:

  First I downloaded each music file by mp3 from youtube. Then I split all the files to 30 second '.wav' files and labeled them by genre. I had 9 genres but with training I have realized that it wasn't necessary and combined the unnecessary genres and ended with 6 results.

### Data Preprocessing:

  With the help of https://github.com/rosariomoscato, and https://github.com/sawan16 's code, I've processed all the data, generating the audio files to spectrum and then to mfcc with a segment of 10 (10_data.json). 
  
 ### Model Training:
 
  Using tensorflow and keras, I've chosen to use two moedls; ANN (Artificial neural network), and CNN (Convolutional Neural Network). The two models had a slight difference of around 20%. 
\n CNN test accuracy: 0.65
\n ANN test accuracy:  0.84
So I had chosen ANN model. 

### Dashboard:

  With the app you can either upload a file ('.mp3', or '.wav') or input a youtube url. When submitted you'll be able to hear the file. If and when there is an error when submitting the url, it's because youtube put an age restriction on it which will make it hard for some videos to be processed.  

Link to the app: https://sarazoloo-mongolian-music-genreclassgenre-classifier-app-w4nzlh.streamlit.app/
