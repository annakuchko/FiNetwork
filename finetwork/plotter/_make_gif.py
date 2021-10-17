import glob
import datetime
from PIL import Image
import shutil

def _mgif(gif_name=None, duration=1000, path=None):
    if path == None:
        path = './tmp/'
        delete = True
    else:
        path = './' + path + '/'
        delete = False
        
    if gif_name == None:
        gif_name = str(datetime.datetime.now().time()).replace(':', '_')
    frames = []
    imgs = sorted(glob.glob(f"{path}/*.png"))
    for j in imgs:
        new_frame = Image.open(j)
        frames.append(new_frame) 
    frames[0].save(
        f'{gif_name}.gif', 
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration, 
        loop=0
        )
    if delete:
        shutil.rmtree(path, ignore_errors=True)
        