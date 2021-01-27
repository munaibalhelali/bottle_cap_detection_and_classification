import json
import os
# import sys
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2 as cv
import numpy as np
import itertools
import imutils

counter = itertools.count(121,1)

next_img = True 
main_window = "Main window" #window name
cv.namedWindow(main_window, cv.WINDOW_NORMAL) # create input window 
cv.namedWindow("Croped image", cv.WINDOW_NORMAL)
sample_points = []
curr_image = [] 
def get_sample(p1, p2):
    croped_img = curr_image[p1[1]:p2[1], p1[0]:p2[0]].copy()
    cv.imshow('Croped image', croped_img)
    # key = cv.waitKey(40) 
     
    croped_img = imutils.resize(croped_img, width = 50, height= 50 )
    cv.imwrite("../data/training_dataset/negative/"+str(next(counter))+".png", croped_img)

def mouseClick(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:            
        sample_points.append((x,y))
        if len(sample_points) == 2:
            get_sample(sample_points[0], sample_points[1])
            sample_points.clear()
            return 

cv.setMouseCallback(main_window, mouseClick)


if __name__ == '__main__':
    imgs= []
    for i in range(1,30):
        imgs.append('../data/dataset/CV20_image_'+str(i)+'.png' )
    next_img = True
    for img in imgs:
        if next_img:
            next_img = False
            image = cv.imread(img)
            curr_image = image.copy()                    
            # sized_image = imutils.resize(image, width = 600, height= 300 )
            cv.imshow(main_window, image)
        key = cv.waitKey(0) 
        if (key == ord(' ')):
            next_img = True
    cv.destroyAllWindows()
