import json
import os
# import sys
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2 as cv
import numpy as np
import base64
def visualize_lables(labels):
    for jsn in labels:
        with open('../data/dataset/'+jsn, 'r') as f:
            data = json.load(f)
        im = cv.imread('../data/dataset/'+data['imagePath'])
        data['imageHeight'] = im.shape[0]
        data['imageWidth'] = im.shape[1]
        data['imageData'] = str(base64.b64encode(im))
        for shape in data['shapes']:
            shape['shape_type'] = 'polygon'
            shape['group_id'] = 'null'
            
        with open('../data/dataset/CV20_image_'+jsn.split('_')[-1], 'w') as outF:
            json.dump(data, outF)
dataset = os.listdir('../data/dataset')
json_files = [f for f in dataset if '.json' in f]

if __name__ == '__main__':
    visualize_lables(json_files) 
