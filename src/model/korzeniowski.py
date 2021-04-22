import tensorflow as tf

properties = {
    'spectrogram_length': 15,
    'spectrogram_step': 5,
    'frame_length': 8192,
    'frame_step': 4410,
    'number_of_filters': 105,
    'number_of_filters_per_octave': 24,
    'filterbank_base_frequency': 65
}


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3),
                               padding='same',
                               input_shape=(properties['spectrogram_length'],
                                            properties['number_of_filters'],
                                            1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D((1, 2)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(64, (3, 3), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D((1, 2)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(128, (9, 12), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(25, (1, 1), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AvgPool2D((3, 13)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Softmax()
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    return model
