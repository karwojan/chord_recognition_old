from pytube import YouTube
from multiprocessing import Pool
import json
import os


def download_song(song):
    print('Downloading "{}"'.format(song['song_name']))
    youtube = YouTube(song['video_id'])
    stream = youtube.streams.filter(only_audio=True, subtype='mp4').order_by('abr').desc().first()
    stream.download(
            os.path.split(song['path_to_annotation'])[0],
            os.path.split(song['path_to_annotation'])[1][:-4])
    print('_____________Completed "{}"'.format(song['song_name']))


with open('youtube_videos_to_download.json', 'r') as f:
    songs = json.load(f)
    songs = list(filter(lambda song: not os.path.isfile(song['path_to_annotation'][:-4] + '.mp4') and 'video_id' in song, songs))

with Pool(8) as p:
    p.map(download_song, songs)
