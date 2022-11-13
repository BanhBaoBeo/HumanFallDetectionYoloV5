import numpy as np 
from imageio import imread,imwrite
import cv2
import os

def transform_image(folder_path, destination_path):
  file_list = os.listdir(folder_path)
  for file_name in file_list:
    img=imread(os.path.join(folder_path , file_name))
    print('Processing ' + file_name)
    b = cv2.flip(img, 1)
    imwrite(os.path.join(destination_path , os.path.splitext(file_name)[0] + '_.png'), b)

transform_image('data', 'processed_data')