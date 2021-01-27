# import sys

# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2 as cv
import imutils
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
import os
import joblib
from sklearn import svm
def read_and_generate_hog(img_path):
    img = cv.imread(img_path)
    resized_image = cv.resize(img, (64, 128))
    fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
    cv.imwrite('../data/hog_imgs/'+'/'.join(img_path.split('/')[-2:]), hog_image)
    return fd, hog_image

if __name__ == '__main__':
    face_up_imgs = os.listdir('../data/training_dataset/positive/face_up')
    face_down_imgs = os.listdir('../data/training_dataset/positive/face_down')
    deformed_imgs = os.listdir('../data/training_dataset/positive/deformed')
    negative_imgs = os.listdir('../data/training_dataset/negative')
    
    samples = []
    labels = []

    for img in face_up_imgs:
        path = '../data/training_dataset/positive/face_up/'+img
        fd, _ = read_and_generate_hog(path)
        samples.append(fd)
        labels.append(1)

    for img in face_down_imgs:
        path = '../data/training_dataset/positive/face_down/'+img
        fd, _ = read_and_generate_hog(path)
        samples.append(fd)
        labels.append(0)
    
    for img in deformed_imgs:
        path = '../data/training_dataset/positive/deformed/'+img
        fd, _ = read_and_generate_hog(path)
        samples.append(fd)
        labels.append(0)

    for img in negative_imgs:
        path = '../data/training_dataset/negative/'+img
        fd, _ = read_and_generate_hog(path)
        samples.append(fd)
        labels.append(0)

    
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(samples), labels, test_size=0.20, random_state=42)
    # Train linear SVM
    print(" Training Linear SVM classifier...")
    model = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(trainData, trainLabels)
    # Evaluate the classifier
    print(" Evaluating classifier on test data ...")
    predictions = model.predict(testData)
    print(classification_report(testLabels, predictions))

    # Save the model:
    joblib.dump(model, 'bottle_cap_svm_model.npy')