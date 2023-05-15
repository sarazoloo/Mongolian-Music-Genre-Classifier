# Mongolian-Music-Genre-Classifier

#### This is a simple music classifier with around 6000 audio files and 6 different genres.
Each audio file is 30 seconds long. The 6 genres are:

Pop, mpop, rock, indie, folk, and hiphop. Although there are a lot of genres mongolian most songs fall into the genres pop, and hiphop. If you are wondering about what mpop is it's mongolian pop, slightly different from the international pop music your used to.

The model is based on the mfcc put in 10 different segments. Here is sample of how each genre's MFCCs look like:


#### Why we use MFCC:
MFCCs divide the audio signal into short time frames and applying the Fourier transform to each frame. Its also becomes non-sensitive to small changes and variations in the spectrum. MFCCs extract relevant features from the audio that are differentiate the different music genres. This is why we want the MFCC feature.
