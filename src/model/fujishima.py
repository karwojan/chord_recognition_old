import tensorflow as tf

properties = {
    'spectrogram_length': 4,
    'spectrogram_step': 4,
    'frame_length': 4096,
    'frame_step': 2205,
    'number_of_filters': 36 * 7,
    'number_of_filters_per_octave': 36,
    'filterbank_base_frequency': 65
}


class PCP(tf.keras.layers.Layer):
    def __init__(self, number_of_filters_per_octave):
        super(PCP, self).__init__()
        self.number_of_filters_per_octave = number_of_filters_per_octave

    def call(self, inputs):
        x = tf.reshape(inputs, [len(inputs), inputs.shape[1], -1, 12,
                                self.number_of_filters_per_octave // 12])
        x = tf.reduce_sum(x, -1)
        x = tf.reduce_sum(x, -2)
        x = tf.reduce_sum(x, -2)
        x = x / tf.reshape(tf.reduce_max(x, -1), (-1, 1))
        return x


class NearestNeighbourCTT(tf.keras.layers.Layer):
    def __init__(self):
        super(NearestNeighbourCTT, self).__init__()
        self.CTT = [
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ]

    def call(self, inputs):
        x = tf.repeat(inputs, 24, -2)
        x = tf.reshape(x, [-1, 24, 12])
        x = (x - self.CTT)**2
        x = tf.reduce_sum(x, -1)
        labels = tf.argmin(x, -1) + 1
        return tf.one_hot(labels, 25)


def create_model():
    return tf.keras.Sequential([
        PCP(properties['number_of_filters_per_octave']),
        NearestNeighbourCTT()
    ])
