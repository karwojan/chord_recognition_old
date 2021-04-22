import tensorflow as tf
import datetime
from preprocessing.load_data import create_dataset
from preprocessing.annotation import create_labels


def _calculate_class_weight(train_audio_files):
    labels = []
    for audio_file in train_audio_files:
        labels.append(
            create_labels(
                tf.strings.regex_replace(audio_file, '\\.wav$', '.lab')))
    labels = tf.concat(labels, axis=0)
    labels = tf.stack((labels[:, 1] - labels[:, 0], labels[:, 2]), axis=1)
    class_weight = {}
    maj_chord_time = tf.reduce_sum(labels[tf.logical_and(
        labels[:, 1] >= 1, labels[:, 1] <= 12)][:, 0])
    min_chord_time = tf.reduce_sum(labels[tf.logical_and(
        labels[:, 1] >= 13, labels[:, 1] <= 24)][:, 0])
    for i in range(0, 13):
        class_weight[i] = 1
    for i in range(13, 25):
        class_weight[i] = (maj_chord_time / min_chord_time).numpy()
    return class_weight


def train(model, model_name, train_audio_files, preprocessing_properties):
    # prepare datasets
    validation_ds = create_dataset(train_audio_files[:len(train_audio_files) //
                                                     10],
                                   **preprocessing_properties,
                                   labels_for_spectrograms=True,
                                   extra_data=False,
                                   batch_size=32).cache()
    train_ds = create_dataset(train_audio_files[len(train_audio_files) // 10:],
                              **preprocessing_properties,
                              labels_for_spectrograms=True,
                              extra_data=True,
                              batch_size=32).cache()
    class_weight = _calculate_class_weight(train_audio_files)

    # train model
    model.fit(train_ds,
              epochs=7,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_sparse_categorical_accuracy',
                      mode='max',
                      patience=2,
                      restore_best_weights=True),
                  tf.keras.callbacks.TensorBoard(
                      log_dir="logs/" + model_name + "_" +
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                  tf.keras.callbacks.ModelCheckpoint(
                      filepath='./checkpoints/' + model_name + "_" +
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                      save_weights_only=True)
              ],
              validation_data=validation_ds,
              class_weight=class_weight)
