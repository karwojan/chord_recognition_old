import os
import glob

audio_file_glob = r'./datasets/**/*.mp4'

for audio_file in glob.glob(audio_file_glob, recursive=True):
    if not os.path.isfile(audio_file[:-3] + 'wav'):
        os.system('ffmpeg -i "' + audio_file + '" -ac 1 "' + audio_file[:-3] + 'wav"')
