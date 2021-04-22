import tensorflow as tf

sample_rate = 44100


def _labels_from_predictions(predictions, spectrogram_duration, time_step):
    centers = tf.linspace(spectrogram_duration / 2,
                          (len(predictions) - 1) * time_step +
                          spectrogram_duration / 2, len(predictions))
    labels = tf.stack((centers - time_step / 2, centers + time_step / 2,
                       tf.cast(tf.argmax(predictions, 1), tf.float32)), 1)
    labels = tf.concat(([[0, labels[0, 0], labels[0, 2]]], labels, [[
        labels[-1, 1],
        (len(predictions) - 1) * time_step + spectrogram_duration, labels[-1,
                                                                          2]
    ]]), 0)
    return labels


def _sum_correctly_predicted_time(labels_pred, labels_true):
    time_points = tf.concat(
        (tf.stack((labels_pred[:, 0], labels_pred[:, 2],
                   tf.zeros((len(labels_pred), )) - 1), 1),
         tf.stack((labels_true[:, 0], tf.zeros(
             (len(labels_true), )) - 1, labels_true[:, 2]),
                  1), [[labels_true[-1, 1], -1, -1]]), 0)
    time_points = tf.gather(time_points, tf.argsort(time_points, 0)[:, 0])
    pred, true = time_points[0, 1:]
    suma = 0
    for time_point, next_time_point in zip(time_points[:-1], time_points[1:]):
        if time_point[1] != -1:
            pred = time_point[1]
        if time_point[2] != -1:
            true = time_point[2]
        if pred == true:
            suma = suma + (next_time_point[0] - time_point[0])
    return suma


def csr(model, dataset, spectrogram_length, spectrogram_step, frame_step):
    correctly_predicted_duration = 0
    total_duration = 0
    for spectrograms, labels_true in dataset:
        labels_pred = _labels_from_predictions(
            model(spectrograms), spectrogram_length * frame_step / sample_rate,
            spectrogram_step * frame_step / sample_rate)
        correctly_predicted_duration = correctly_predicted_duration + _sum_correctly_predicted_time(
            labels_pred, labels_true)
        total_duration = total_duration + labels_true[-1, 1]
    return correctly_predicted_duration / total_duration


if __name__ == '__main__':
    labels_true = tf.constant([[0, 0.7, 0], [0.7, 1.5, 1], [1.5, 3, 2]])
    predictions = tf.constant([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                               [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    def model(spectrograms):
        return predictions

    print("labels_pred", _labels_from_predictions(predictions, 1, 0.333))
    print("labels_true", labels_true)
    print(
        "sum",
        _sum_correctly_predicted_time(
            _labels_from_predictions(predictions, 1, 0.333), labels_true))
    print(
        "csr",
        csr(model, tf.data.Dataset.from_tensors(("spectrograms", labels_true)),
            3, 1, 44100 / 3))
