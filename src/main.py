import sys

sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2 as cv
import imutils
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.feature import hog

from frame_filter import *
from color import *


if __name__ == '__main__':
    imgs= []
    for i in range(1,100):
        imgs.append('../data/dataset/CV20_image_'+str(i)+'.png' )
    next = True
    for img in imgs[:7]:
        if next:
            next = False
            print(img)
            image = cv.imread(img)
            sized_image = imutils.resize(image, width = 600, height= 300 )
            sub_image = sized_image[100:200,250:350]
            sized_image =cv.GaussianBlur(sized_image, (3, 3), 0)
            lap = cv.Laplacian(sized_image,cv.CV_64F,ksize=3) 
            lap = np.uint8(np.absolute(lap))
            lap_cpy = lap.copy()
            
            circles = cv.HoughCircles(cv.cvtColor(lap_cpy, cv.COLOR_BGR2GRAY),cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=1,maxRadius=20)
    #         print(circles)
            circles = np.uint16(np.around(circles))
            
            for i in circles[0,:]:
                # draw the outer circle
            #     print(i)
                cv.circle(lap_cpy,(i[0],i[1]),i[2],(255,0,0),2)
                # draw the center of the circle
                cv.circle(lap_cpy,(i[0],i[1]),2,(255,0,0),3)

            gray  = cv.cvtColor(lap, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray,100,200)
            fltrd_edges = fltrFrame(edges)
            
            # Otsu's thresholding after Gaussian filtering
            blur = cv.GaussianBlur(edges,(5,5),0)
            ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            cv.imshow('Threshold', th3 )
            cv.imshow('fltrd edges', fltrd_edges)
            new_img = np.zeros(edges.shape)
            new_img = cv.merge((new_img, new_img, new_img))


            edges = cv.merge((edges, edges, edges))
            cv.imshow('lap', np.hstack((lap_cpy,edges, sized_image)))
            
            output = cv.bitwise_and(sized_image, sized_image, mask = fltrd_edges)
            
    #         cv.rectangle(sized_image, (250, 100), (350,200), (int(bg[0]), int(bg[1]), int(bg[2])), -1)
            cv.imshow('origianl', np.hstack([sized_image, output]))

        key = cv.waitKey(0) & 0xFF 

        if (key == ord(' ')):
    #         break
            next = True
    #     time.sleep(0.5)
    cv.imwrite('../data/proccessed_img.png', lap_cpy)
    cv.destroyAllWindows()