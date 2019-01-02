import os
import glob
from moviepy.editor import *

base_dir = os.path.realpath("./images")
print(base_dir)

gif_name = 'pic'
fps = 24

file_list = glob.glob('samples/dfc_vae3_bak/for_anim/*.png')  # Get all the pngs in the current directory
file_list_sorted = sorted(file_list,reverse=False)  # Sort the images

clips = [ImageClip(m).set_duration(0.1)
         for m in file_list_sorted[:120]]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("test.mp4", fps=fps)
