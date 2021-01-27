import json
import os
# import sys
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2 as cv
import numpy as np
area_list = []
x_list = []
y_list = []
def visualize_lables(labels):
    for jsn in labels[:10]:
        with open('../data/dataset/'+jsn, 'r') as f:
            data = json.load(f)
        im = cv.imread('../data/dataset/'+data['imagePath'])
        shapes = data['shapes']
        for points in [shape['points'] for shape in shapes]:
            points = np.uint64(points)    
            cv.polylines(im, [points], True, (0, 0, 255), 3 )
            x, y, w , h = cv.boundingRect(points)

            area = w*h  
            area_list.append(area)
            x_list.append(w)
            y_list.append(h)
        cv.imshow('image', im)
        cv.waitKey(0)
    cv.destroyAllWindows()

dataset = os.listdir('../data/dataset')
json_files = [f for f in dataset if '.json' in f]

if __name__ == '__main__':
    visualize_lables(json_files) 
    print('maximum area: ',max(area_list))
    print('minimum area: ',min(area_list))
    print('maximum x: ',max(x_list))
    print('minimum x: ',min(x_list))
    print('maximum y: ',max(y_list))
    print('minimum y: ',min(y_list))