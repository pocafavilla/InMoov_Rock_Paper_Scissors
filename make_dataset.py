import numpy as np
import os
from PIL import Image

class MakeDataset():

    def read(self,subdir):
        img_type = '.jpg'
        dir = os.path.join('/localdata/Leona_Maehler/practice/InMoov/t2/data/', subdir)
        imgs = []

        for file_name in os.listdir(dir):
            if file_name[-4:] == img_type:
                img = np.asarray(Image.open(os.path.join(dir,file_name))) #/255
                imgs.append(img)

        return np.asarray(imgs)


    def read_all(self, end):

        all_imgs = [list(),list(),list()]#0 = rock; 1 = paper; 2 = scissors
        for i, subdir in enumerate(os.listdir("/localdata/Leona_Maehler/practice/InMoov/t2/data/")):
            if i>= end:
                for e in range(len(all_imgs)):
                    all_imgs[e] = np.asarray(all_imgs[e])
                return all_imgs
            for j, subsubdir in enumerate(['c0','c1','c2']):           
                if os.path.isdir(os.path.join('/localdata/Leona_Maehler/practice/InMoov/t2/data/', 
                    os.path.join(subdir,subsubdir))):

                    res = self.read(os.path.join(subdir,subsubdir))
                    all_imgs[j].extend(res)
            




