import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
import librosa
import cv2

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
        print("write WAV data resampled successfully!")
    else : return resampled_array

def process_wav_to_image(dir_wav:str, dir_tar:str,visual = False, output ="write"):
    Sample_rate, audio_data = wavfile.read(dir_wav)
    audio_mono_data = audio_data[:,0]
    # print("pass")
    audio_mono_data = audio_mono_data.astype('int')/ 32767
    spectrogram = librosa.feature.melspectrogram(y=audio_mono_data,sr=Sample_rate)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
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









