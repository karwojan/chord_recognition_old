import tensorflow as tf
from preprocessing.annotation import create_labels_for_spectrograms, create_labels
from preprocessing.spectrogram import create_spectrograms


@tf.function
def load_single_song(audio_file, spectrogram_length, spectrogram_step,
                     frame_length, frame_step, number_of_filters,
                     number_of_filters_per_octave, filterbank_base_frequency,
                     labels_for_spectrograms, extra_data, batch_size):
    raw_file = tf.io.read_file(audio_file)
    samples, sample_rate = tf.audio.decode_wav(raw_file)
    spectrograms = create_spectrograms(
        samples,
        sample_rate,
        spectrogram_length=spectrogram_length,
        spectrogram_step=spectrogram_step,
        frame_length=frame_length,
        frame_step=frame_step,
        number_of_filters=number_of_filters,
        number_of_filters_per_octave=number_of_filters_per_octave,
        filterbank_base_frequency=filterbank_base_frequency,
        extra_data=extra_data)
    spectrograms.set_shape((None, spectrogram_length, number_of_filters, 1))
    number_of_spectrograms = len(spectrograms)
    if extra_data:
        number_of_spectrograms /= 12
    if labels_for_spectrograms:
        labels = create_labels_for_spectrograms(
            tf.strings.regex_replace(audio_file, '\\.wav$', '.lab'),
            number_of_spectrograms=number_of_spectrograms,
            spectrogram_duration=spectrogram_length * frame_step /
            tf.cast(sample_rate, tf.float32),
            time_step=spectrogram_step * frame_step /
            tf.cast(sample_rate, tf.float32),
            extra_data=extra_data)
        labels.set_shape((None))
    else:
        labels = create_labels(
            tf.strings.regex_replace(audio_file, '\\.wav$', '.lab'))
        labels.set_shape((None, 3))
    if batch_size is None:
        return tf.data.Dataset.from_tensors((spectrograms, labels))
    else:
        spectrograms_ds = tf.data.Dataset.from_tensor_slices(spectrograms)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip(
            (spectrograms_ds, labels_ds)).batch(batch_size)


def create_dataset(audio_files, spectrogram_length, spectrogram_step,
                   frame_length, frame_step, number_of_filters,
                   number_of_filters_per_octave, filterbank_base_frequency,
                   labels_for_spectrograms, extra_data, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(audio_files)
    ds = ds.flat_map(lambda audio_file: load_single_song(
        audio_file, spectrogram_length, spectrogram_step, frame_length,
        frame_step, number_of_filters, number_of_filters_per_octave,
        filterbank_base_frequency, labels_for_spectrograms, extra_data,
        batch_size))
    ds = ds.prefetch(2)
    return ds


if __name__ == '__main__':
    properties = {
        'spectrogram_length': 30,
        'spectrogram_step': 15,
        'frame_length': 4096,
        'frame_step': 2048,
        'number_of_filters': 5 * 36,
        'number_of_filters_per_octave': 36,
        'filterbank_base_frequency': 65
    }
    data = create_dataset(['test_datasets/13_-_Yesterday.wav'],
                          **properties,
                          labels_for_spectrograms=True,
                          extra_data=False,
                          batch_size=32)
