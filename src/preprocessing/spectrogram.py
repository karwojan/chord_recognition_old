import tensorflow as tf
import matplotlib.pyplot as plt


@tf.function
def create_spectrograms(samples,
                        sample_rate,
                        spectrogram_length,
                        spectrogram_step,
                        frame_length,
                        frame_step,
                        number_of_filters,
                        number_of_filters_per_octave,
                        filterbank_base_frequency,
                        extra_data=False):
    frames = create_spectrogram(samples, sample_rate, frame_length, frame_step,
                                number_of_filters,
                                number_of_filters_per_octave,
                                filterbank_base_frequency)
    spectrograms = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    number_of_spectrograms = tf.math.ceil(
        (len(frames) - spectrogram_length) / spectrogram_step)
    spectrogram_idx = 0
    for i in tf.range(0, number_of_spectrograms, dtype=tf.int32):
        new_spectrogram = frames[i * spectrogram_step:i * spectrogram_step +
                                 spectrogram_length]
        spectrograms = spectrograms.write(spectrogram_idx, new_spectrogram)
        spectrogram_idx += 1
        if extra_data:
            for i in tf.range(1, 12):
                shift = tf.cast(i * number_of_filters_per_octave / 12,
                                tf.int32)
                shifted_spectrogram = tf.expand_dims(
                    tf.squeeze(tf.roll(new_spectrogram, shift, 1)) *
                    tf.concat([
                        tf.zeros([shift]),
                        tf.ones([number_of_filters - shift])
                    ], 0), -1)
                spectrograms = spectrograms.write(spectrogram_idx,
                                                  shifted_spectrogram)
                spectrogram_idx += 1

    return spectrograms.stack()


@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.float32)
])
def create_spectrogram(samples, sample_rate, frame_length, frame_step,
                       number_of_filters, number_of_filters_per_octave,
                       filterbank_base_frequency):
    samples = tf.squeeze(samples)
    frames = tf.signal.stft(samples, frame_length, frame_step)
    frames = tf.abs(frames)
    filterbank = generate_filterbank(number_of_filters,
                                     number_of_filters_per_octave,
                                     filterbank_base_frequency,
                                     frame_length // 2 + 1, sample_rate // 2)
    frames = tf.math.log(1 + tf.matmul(frames, tf.transpose(filterbank)))
    frames = tf.expand_dims(frames, -1)
    return frames


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
])
def generate_filterbank(number_of_filters, number_of_filters_per_octave,
                        base_frequency, filter_length, frequency_range):
    filters = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(0, number_of_filters, dtype=tf.int32):
        filters = filters.write(
            i,
            generate_filter(i, number_of_filters_per_octave, base_frequency,
                            filter_length, frequency_range))
    return filters.stack()


@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32)
])
def generate_filter(number_of_filter, number_of_filters_per_octave,
                    base_frequency, filter_length, frequency_range):
    triangle = tf.stack(
        [number_of_filter - 1, number_of_filter, number_of_filter + 1])
    triangle = tf.cast(base_frequency, tf.float64) * (2**(
        triangle / number_of_filters_per_octave))
    triangle = tf.cast(
        tf.round(triangle / tf.cast(frequency_range, tf.float64) *
                 tf.cast(filter_length, tf.float64)),
        dtype=tf.int32)
    if triangle[0] == triangle[1]:
        triangle = triangle - [1, 0, 0]
    if triangle[2] == triangle[1]:
        triangle = triangle + [0, 0, 1]
    return tf.concat([
        tf.zeros(triangle[0]),
        tf.cast(tf.linspace(0, 1, triangle[1] - triangle[0]),
                dtype=tf.float32),
        tf.cast(tf.linspace(1, 0, triangle[2] - triangle[1]),
                dtype=tf.float32),
        tf.zeros(filter_length - triangle[2])
    ], 0)


if __name__ == '__main__':

    def plot_spectrogram(spectrogram):
        plt.pcolormesh(tf.transpose(tf.squeeze(spectrogram)))
        plt.show()

    # filterbank test
    print('Generating filterbank...')
    filters = generate_filterbank(36 * 5, 36, 65, 4096 // 2 + 1, 22050)
    for filtr in filters:
        plt.plot(filtr)
    plt.xlim(0, 2100)
    plt.show()

    # spectrogram test
    print('Generating spectrogram')
    raw_file = tf.io.read_file('test_datasets/13_-_Yesterday.wav')
    samples, sample_rate = tf.audio.decode_wav(raw_file)
    spectrogram = create_spectrogram(samples, sample_rate, 4096, 2048, 36 * 5,
                                     36, 65)
    frames_per_second = 44100 // 2048
    plot_spectrogram(spectrogram[5 * frames_per_second:10 * frames_per_second])

    # spectrograms test
    print('Generating spectrograms')
    spectrograms = create_spectrograms(samples, sample_rate, 15, 15, 4096,
                                       2048, 36 * 5, 36, 65, True)
    for spectrogram in spectrograms[:13]:
        plot_spectrogram(spectrogram)
