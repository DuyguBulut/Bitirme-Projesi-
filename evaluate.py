# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:32:10 2020

@author: duygu
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import operator
import random
import glob
import os.path
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

BATCH_SIZE = 120
SEED_VAL = 100


test_dir  = "C:\\Users\duygu\\Desktop\\testSet"
alphabet_classes = ['A','B','C','Ç','D','E','F','G','Ğ','H','I','İ','J','K','L',
           'M','N','O','Ö','P','R','S', 'Ş' ,'T','U','Ü','V', 'Y','Z']



def get_test_generator():
    
    datagen = ImageDataGenerator(
        rescale=1./255)

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        classes=alphabet_classes,
        class_mode='categorical',
        seed = SEED_VAL)

    return test_generator


def main():

    test_generator = get_test_generator()
        
    model = load_model('C:\\Users\\duygu\\Desktop\\design.005-0.00.hdf5')

    score = model.evaluate_generator(test_generator, verbose=1)
    
    print("loss:", score[0], "acc:", score[1])

if __name__ == '__main__':
    main()
