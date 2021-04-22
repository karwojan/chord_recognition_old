import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from time import time
from preprocessing.spectrogram import create_spectrogram

chords = [
    'N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B', 'H', 'c',
    'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'b', 'h'
]


def analyse(model, preprocessing_properties):
    # prepare buffer
    sample_rate = 44100
    buff_len = 5 * sample_rate
    buff = tf.zeros((buff_len, ), dtype=tf.float32)

    # prepare audio stream
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=sample_rate // 20)

    # prepare plot
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(tf.scatter_nd(((0, 0), ), (1, ), (105, 49)))
    text = ax.text(45, 90, "N", fontsize=30, color='red')

    def render(i):
        nonlocal buff
        t = time()
        data = stream.read(stream.get_read_available())
        data = tf.cast(tf.io.decode_raw(data, tf.int16), dtype=tf.float32)
        buff = tf.concat((buff[len(data):], data), 0)
        print("Czytanie, dekodowanie i bufforowanie", time() - t)

        t = time()
        spectrogram = create_spectrogram(
            tf.reshape(buff, (-1, 1)),
            sample_rate,
            frame_length=preprocessing_properties['frame_length'],
            frame_step=preprocessing_properties['frame_step'],
            number_of_filters=preprocessing_properties['number_of_filters'],
            number_of_filters_per_octave=preprocessing_properties[
                'number_of_filters_per_octave'],
            filterbank_base_frequency=preprocessing_properties[
                'filterbank_base_frequency'])
        print("Tworzenie spektrogramu", time() - t)

        t = time()
        spectrogram_fragment = spectrogram[
            -preprocessing_properties['spectrogram_length']:]
        chord = chords[tf.argmax(
            model(
                tf.reshape(
                    spectrogram_fragment,
                    (1, preprocessing_properties['spectrogram_length'],
                     preprocessing_properties['number_of_filters'], 1)))[0])]
        text.set_text(chord)
        print("Rozpoznawanie akordu", time() - t, chord)

        spectrogram = tf.reshape(tf.transpose(tf.squeeze(spectrogram)), (-1, ))
        mesh.set_array(spectrogram / tf.reduce_max(spectrogram))
        return mesh, text

    animation = FuncAnimation(fig, render, interval=100, blit=True)
    plt.show()
