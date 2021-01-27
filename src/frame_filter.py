import cv2 as cv
def fltrFrame (Mask,Thresh_min=100,Thresh_max=255,Thresh_type=cv.THRESH_BINARY,Kernel_size=(3,3)):
    retval,fgthres = cv.threshold(Mask, Thresh_min, Thresh_max, Thresh_type)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, Kernel_size)

    # Fill any small holes
#     closing = cv.morphologyEx(fgthres, cv.MORPH_CLOSE, kernel, iterations = 2)
    # Remove noise
#     opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv.dilate(fgthres, kernel, iterations = 2)

    return dilation 