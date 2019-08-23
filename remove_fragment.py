import os
dirtocheck = './../../audio_chunks/dev/S09'
for root, _, files in os.walk(dirtocheck):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.getsize(fullpath) < 900 * 1024:
            os.remove(fullpath)
