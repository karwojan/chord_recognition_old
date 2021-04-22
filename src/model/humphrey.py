import tensorflow as tf

properties = {
    'spectrogram_length': 100,
    'spectrogram_step': 20,
    'frame_length': 4096,
    'frame_step': 2205,
    'number_of_filters': 252,
    'number_of_filters_per_octave': 36,
    'filterbank_base_frequency': 27.5
}


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.AveragePooling2D(
            (5, 1),
            input_shape=(properties['spectrogram_length'],
                         properties['number_of_filters'], 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (6, 25), padding='valid',
                               activation='tanh'),
        tf.keras.layers.MaxPool2D((1, 3)),
        tf.keras.layers.Conv2D(20, (6, 27), padding='valid',
                               activation='tanh'),
        tf.keras.layers.Conv2D(24, (6, 27), padding='valid',
                               activation='tanh'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(25)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    return model
