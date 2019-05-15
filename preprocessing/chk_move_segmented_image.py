from keras.models import load_model
from keras.preprocessing import image
import os
import shutil
import numpy as np

model = load_model('../trainedModels/segmentation_verification/segmentation_verification_model.h5')
source_dir = '../../../datasets/All/segmented'
dest_dir = '../../../datasets/All/augmented'
for a in os.listdir(source_dir):
    for b in os.listdir(source_dir+'/'+a):
        for c in os.listdir(source_dir+'/'+a+'/'+b):
                        img = image.load_img(source_dir+'/'+a+'/'+b+'/'+c, target_size=(150, 150))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = x/255
                        preds = model.predict(x)
						#if segmentation is good then move the image for training 
                        if (preds[0][0])>(preds[0][1]):
                            shutil.move(source_dir+'/'+a+'/'+b+'/'+c, dest_dir+'/'+a+'/'+b+'/'+c)
                            print(source_dir+'/'+a+'/'+b+'/'+c+'  moved to  '+dest_dir+'/'+a+'/'+b+'/')
                


