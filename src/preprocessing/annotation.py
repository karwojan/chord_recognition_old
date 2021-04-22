import tensorflow as tf
import lark


class Chord:

    _parser = lark.Lark('''\
start: pitchname ":" shorthand [ "(" ilist ")" ] [ "/" bass ]
        | pitchname ":" "(" ilist ")" [ "/" bass ]
        | pitchname [ "/" bass ]
        | NO_CHORD
pitchname: NATURAL MODIFIER*
shorthand: SHORTHAND
ilist: ilist_element ( "," ilist_element ) *
ilist_element: [ STAR ] interval
bass: interval
interval: MODIFIER* degree
degree: NON_ZERO_DIGIT DIGIT*
STAR: "*"
NATURAL: "A" | "B" | "C" | "D" | "E" | "F" | "G"
MODIFIER: "b" | "#"
NON_ZERO_DIGIT: "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
DIGIT: "0" | NON_ZERO_DIGIT
SHORTHAND: "minmaj7" | "maj7" | "min7" | "dim7" | "hdim7" 
            | "maj6" | "min6" | "maj9" | "min9" | "sus2" | "sus4" 
            | "maj" | "min" | "dim" | "aug" | "7" | "9"
NO_CHORD: "N"''')
    _naturals = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    _modifiers = {'b': -1, '#': +1}
    _shorthands = {
        'maj': {'1', '3', '5'},
        'min': {'1', 'b3', '5'},
        'dim': {'1', 'b3', 'b5'},
        'aug': {'1', '3', '#5'},
        'maj7': {'1', '3', '5', '7'},
        'min7': {'1', 'b3', '5', 'b7'},
        '7': {'1', '3', '5', 'b7'},
        'dim7': {'1', 'b3', 'b5', 'bb7'},
        'hdim7': {'1', 'b3', 'b5', 'b7'},
        'minmaj7': {'1', 'b3', '5', '7'},
        'maj6': {'1', '3', '5', '6'},
        'min6': {'1', 'b3', '5', '6'},
        '9': {'1', '3', '5', 'b7', '9'},
        'maj9': {'1', '3', '5', '7', '9'},
        'min9': {'1', 'b3', '5', 'b7', '9'},
        'sus2': {'1', '2', '5'},
        'sus4': {'1', '4', '5'}
    }

    def __init__(self, annotation):
        chord_tree = Chord._parser.parse(annotation)
        if isinstance(chord_tree.children[0], lark.lexer.Token):
            self.nochord = True
        else:
            self.nochord = False
            # root
            self.root = Chord._naturals[chord_tree.children[0].children[0]]
            for modifier in chord_tree.children[0].children[1:]:
                self.root = (self.root + Chord._modifiers[modifier]) % 12
            # bass
            self.bass = '1'
            for bass in chord_tree.find_data('bass'):
                self.bass = "".join(bass.scan_values(lambda x: True))
            # intervals
            self.intervals = set()
            for shorthand in chord_tree.find_data('shorthand'):
                self.intervals = Chord._shorthands[
                    shorthand.children[0].value].copy()
            for ilist in chord_tree.find_data('ilist'):
                for ilist_element in ilist.find_data('ilist_element'):
                    text = "".join(ilist_element.scan_values(lambda x: True))
                    if text[0] == '*':
                        self.intervals.remove(text[1:])
                    else:
                        self.intervals.add(text)
            if len(self.intervals) == 0:
                self.intervals = Chord._shorthands['maj'].copy()

    def __str__(self):
        return 'root: {self.root}, bass: {self.bass}, intervals: {self.intervals}'.format(
            self=self)


chord_types = [{'1', '3', '5'}, {'1', 'b3', '5'}]


def find_label(annotation):
    annotation = annotation.numpy().decode('ascii')
    chord = Chord(annotation)
    if chord.nochord is True:
        return 0
    matching_chord_types = list(
        filter(lambda chord_type: chord_type <= chord.intervals, chord_types))
    if len(matching_chord_types) == 0:
        return 0
    longest_matching_chord_type = max(matching_chord_types,
                                      key=lambda chord_type: len(chord_type))
    return 1 + 12 * chord_types.index(longest_matching_chord_type) + chord.root


@tf.function
def create_labels(annotation_file):
    raw_file = tf.io.read_file(annotation_file)
    raw_file = tf.strings.strip(raw_file)
    words = tf.reshape(tf.strings.split(raw_file), [-1, 3])
    start_times = tf.strings.to_number(words[:, 0])
    end_times = tf.strings.to_number(words[:, 1])
    labels = tf.map_fn(lambda annotation: tf.py_function(
        find_label, [annotation], tf.float32),
                       words[:, 2],
                       fn_output_signature=tf.float32)
    return tf.stack([start_times, end_times, labels], 1)


@tf.function
def create_labels_for_spectrograms(annotation_file,
                                   number_of_spectrograms,
                                   spectrogram_duration,
                                   time_step,
                                   extra_data=False):
    labels = create_labels(annotation_file)
    labels_for_spectrograms = tf.TensorArray(tf.int32,
                                             size=0,
                                             dynamic_size=True)
    spectrogram_idx = 0
    for i in tf.range(0, number_of_spectrograms, dtype=tf.int32):
        start_time = tf.cast(i, tf.float32) * time_step
        center_time = start_time + spectrogram_duration / 2
        label = tf.cast(labels[labels[:, 0] <= center_time][-1, 2], tf.int32)
        labels_for_spectrograms = labels_for_spectrograms.write(
            spectrogram_idx, label)
        spectrogram_idx += 1
        if extra_data:
            if label == 0:
                extra_labels = tf.zeros([11], dtype=tf.int32)
            elif label < 13:
                extra_labels = ((tf.range(1, 12, dtype=tf.int32) +
                                 (label - 1)) % 12) + 1
            else:
                extra_labels = ((tf.range(1, 12, dtype=tf.int32) +
                                 (label - 13)) % 12) + 13
            labels_for_spectrograms = labels_for_spectrograms.scatter(
                tf.range(11) + spectrogram_idx, extra_labels)
            spectrogram_idx += 11
    return labels_for_spectrograms.stack()


if __name__ == '__main__':
    print(create_labels('test_datasets/13_-_Yesterday.lab'))
    for label in create_labels_for_spectrograms(
            'test_datasets/13_-_Yesterday.lab', 10, 2, 1):
        print(label.numpy())
    for label in tf.reshape(
            create_labels_for_spectrograms('test_datasets/13_-_Yesterday.lab',
                                           10, 2, 1, True), [-1, 12]):
        print(label.numpy())
