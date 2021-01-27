import json
import os
# import sys
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2 as cv
import numpy as np
import itertools
import imutils

face_down_counter = itertools.count(1,1)
face_up_counter = itertools.count(1,1)
deformed_counter = itertools.count(1,1)
def visualize_lables(labels):
    for jsn in labels:
        with open('../data/dataset/'+jsn, 'r') as f:
            data = json.load(f)
        im = cv.imread('../data/dataset/'+data['imagePath'])
        shapes = data['shapes']
        for label, points in [(shape['label'], shape['points']) for shape in shapes]:
            points = np.uint64(points)    
            # cv.polylines(im, [points], True, (0, 0, 255), 3 )
            ## (1) Crop the bounding rect
            rect = cv.boundingRect(points)
            x,y,w,h = rect
            croped = im[y:y+h, x:x+w].copy()

            ## (2) make mask
            mask = np.zeros(croped.shape[:2], np.uint8)

            ## (3) do bit-op
            # bottle_cap = cv.bitwise_and(croped, croped, mask=mask)
            croped = cv.resize(croped, (64, 128 ))
            if label == 'BottleCap_FaceDown':
                cv.imwrite("../data/training_dataset/positive/face_down/"+str(next(face_down_counter))+".png", croped)

            elif label == 'BottleCap_FaceUp':
                cv.imwrite("../data/training_dataset/positive/face_up/"+str(next(face_up_counter))+".png", croped)

            elif label == 'BottleCap_Deformed':
                cv.imwrite("../data/training_dataset/positive/deformed/"+str(next(deformed_counter))+".png", croped)
            # cv.imshow('image', croped)
            # # cv.imshow('image', im)
            # cv.waitKey(0)
    cv.destroyAllWindows()

dataset = os.listdir('../data/dataset')
json_files = [f for f in dataset if '.json' in f]

if __name__ == '__main__':
    visualize_lables(json_files) 

