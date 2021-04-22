import glob
import random

audio_file_glob = r'datasets/**/*.wav'
test_audio_files_storage = 'test_audio_files.txt'
train_audio_files_storage = 'train_audio_files.txt'

audio_files = glob.glob(audio_file_glob, recursive=True)
random.Random(47).shuffle(audio_files)

test_audio_files = audio_files[:len(audio_files) // 3]
train_audio_files = audio_files[len(test_audio_files):]

with open(test_audio_files_storage, 'w') as f:
    for filename in test_audio_files:
        f.write(filename)
        f.write('\n')

with open(train_audio_files_storage, 'w') as f:
    for filename in train_audio_files:
        f.write(filename)
        f.write('\n')
