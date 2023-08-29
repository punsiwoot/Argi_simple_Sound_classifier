import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
import librosa
import cv2
import scipy.io.wavfile as wavfile
from pydub import AudioSegment

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow.keras.layers as nn

def resample_wav(input_file, output_file, new_length, file=True):
    # Open the input WAV file
    with wave.open(input_file, 'rb') as wav_in:
        # Get the audio file's properties
        params = wav_in.getparams()
        num_frames = params.nframes

        # Read the audio data as a byte stream
        audio_data = wav_in.readframes(num_frames)

    # Convert the byte stream to a numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Resample the audio data to the desired length
    resampled_array = signal.resample(audio_array, new_length)

    # Open the output WAV file
    if file:
        with wave.open(output_file, 'wb') as wav_out:
            # Set the output WAV file's properties
            wav_out.setparams(params)

            # Convert the numpy array back to a byte stream
            output_data = resampled_array.astype(np.int16).tobytes()

            # Write the resampled audio data to the output file
            wav_out.writeframes(output_data)
        # print("write WAV data resampled successfully!")
    else : return resampled_array

def process_wav_to_image(dir_wav:str, dir_tar:str,visual = False, output ="write"):
    Sample_rate, audio_data = wavfile.read(dir_wav)
    audio_mono_data = audio_data[:,0]
    # print("pass")
    audio_mono_data = audio_mono_data.astype('float32')#/ 32767
    spectrogram = librosa.feature.melspectrogram(y=audio_mono_data,sr=Sample_rate,n_fft=512,window=512)
    # spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    if visual:
        # print(spectrogram.shape)
        # spectrogram_show = np.expand_dims(spectrogram, axis=-1)
        plt.imshow(spectrogram)
    if output == "write":
        # spectrogram = np.expand_dims(spectrogram, axis=-1)
        # print(spectrogram.shape)
        cv2.imwrite(dir_tar,spectrogram)
    elif output == "get":
        return spectrogram


def get_model_sound():
    model = tf.keras.Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(58, 342,1)),
    nn.MaxPooling2D(pool_size=(2,2)),
    Conv2D(8, (3,3), activation='relu'),
    nn.MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.load_weights("save_w/weight")
    input_signature = [tf.TensorSpec(shape=(1,58,342,1), dtype=tf.int32)]
    model_fn = tf.function(input_signature=input_signature)(model.call)
    return  model_fn

def preprocess_sound(sound_path): #get wav file
    data = process_wav_to_image(dir_wav=sound_path, dir_tar = None, output='get')
    data = data[70:,:]
    data_thres = data>150
    data_thres = data_thres.astype(int)
    return data_thres

def turn_m3a_to_wav(dir:str):
    audio = AudioSegment.from_file(dir, format="m4a")
    audio.export(dir[:-4]+".wav", format="wav")

def get_prediction(model, path):
    data = preprocess_sound(path)
    data = data.astype(int)
    data = tf.expand_dims(data, axis = 0)
    # print(data)
    result = model(data)[0][0]
    return 1 if result > 0.5 else 0

# def compress_sound(path_read,output_file):
    # resample_wav(path_read, output_file, 350000)