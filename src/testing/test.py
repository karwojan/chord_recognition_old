from preprocessing.load_data import create_dataset
from testing.confusion_matrix import calculate_confusion_matrix, render_confusion_matrix
from testing.csr import csr


def test(model, test_audio_files, preprocessing_properties, plot=False):
    # prepare datasets
    test_ds = create_dataset(test_audio_files,
                             **preprocessing_properties,
                             labels_for_spectrograms=True,
                             extra_data=False,
                             batch_size=None)
    test_ds_time_labels = create_dataset(test_audio_files,
                                         **preprocessing_properties,
                                         labels_for_spectrograms=False,
                                         extra_data=False,
                                         batch_size=None)

    # test model
    print('CONFUSION MATRIX')
    confusion_matrix = calculate_confusion_matrix(model, test_ds)
    print(confusion_matrix)
    if plot:
        render_confusion_matrix(confusion_matrix, [
            'N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'B',
            'H', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'b', 'h'
        ])
    print('\n\n\n')

    print("CSR")
    print(
        csr(model, test_ds_time_labels,
            preprocessing_properties['spectrogram_length'],
            preprocessing_properties['spectrogram_step'],
            preprocessing_properties['frame_step']))
    print('\n\n\n')
