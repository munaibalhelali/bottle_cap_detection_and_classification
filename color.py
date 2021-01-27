import cv2 as cv
#TODO need to add the hsv color funciton

def get_color(sub_image):
    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv.split(sub_image)

    features = []
    # loop over the image channels
    max_color=[]
    min_color = []
    hists =[]
    
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        max_min = np.where(np.array(hist)>=300)[0]
        _max = np.max(np.array(max_min))
        _min = np.min(np.array(max_min))

        hists.append(hist)
        
        max_color.append((_min, _max))
        
        features.extend(hist)

    return max_color