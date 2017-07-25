from carnie_helper import RudeCarnie
import os
import random

def load_imgs(img_directory):        
    imgs = []
    for root, subdirs, files in os.walk(img_directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                 imgs.append(os.path.join(root, file))
    
    return imgs

rc = RudeCarnie()

data_dir ='/usr/src/app/assets/thumbnails'
files = load_imgs(data_dir)
files = random.sample(files, 30)

assert len(files) != 0, 'test'
best = rc.get_gender(files)

for comb in zip(files, best):
    print('{} : {}'.format(comb[0], comb[1]))
