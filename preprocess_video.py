import subprocess
from sys import argv
video_file = argv[1]
video_name = video_file.split(".")[0]
subprocess.run(f'ffmpeg -i {video_file}.mp4 -ac 1 {video_name}.mp3',shell=True)