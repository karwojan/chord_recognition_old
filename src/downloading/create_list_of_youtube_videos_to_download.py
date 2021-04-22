import os
import glob
import re
import json

annotations_file_glob = r'./datasets/**/*.lab'
storage = 'youtube_videos_to_download.json'


def album_dir_name_to_album_name(name):
    return re.match(r'^\d+_-_(.*)$', name).group(1).replace('_', ' ')


def song_file_name_to_song_name(name):
    return re.match(r'^.+_-_(.*).lab$', name).group(1).replace('_', ' ')


if os.path.isfile(storage):
    with open(storage, 'r') as f:
        songs = json.load(f)
else:
    songs = [{
            'path_to_annotation': path,
            'album_name': album_dir_name_to_album_name(os.path.split(os.path.split(path)[0])[1]),
            'song_name': song_file_name_to_song_name(os.path.split(path)[1])} 
                for path in glob.glob(annotations_file_glob, recursive=True)]


for song in filter(lambda song: 'video_id' not in song, songs):
    print(song['album_name'], ' --- ', song['song_name'])
    song['video_id'] = input("\tid: ")
    with open(storage, 'w') as f:
        json.dump(songs, f, indent=2)
