import numpy as np 
import joblib
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
#from sklearn.externals import joblib
import cv2

from skimage import color
import pickle
import os
import time 


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


            
#model_path = 'models'
#file= 'models\finalized_model.sav'            
#clf = joblib.load(os.path.join(model_path, 'finalized_model.sav'))
#clf= joblib.load(file)
#clf = pickle.load(open(file, 'rb'))
pkl_filename = "features.pkl"
with open(pkl_filename, 'rb') as file:
    clf = pickle.load(file)

    
files= 'VideoTest\VID_20190717_180242 (2).mp4'
cap = cv2.VideoCapture(files)
#fgbg = cv2.createBackgroundSubtractorMOG2(500,50,True)
while (cap.isOpened()):
    ret,frame=cap.read()
     
    im = imutils.resize(frame, width = min(300, frame.shape[1]))
    """
    ycrcb= cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y,cr,cb= cv2.split(ycrcb)
     
    channel_y_eq=cv2.equalizeHist(y)
    channel_cr_eq=cv2.equalizeHist(cr)
    channel_cb_eq=cv2.equalizeHist(cb)
    
    eq_ycrcb= cv2.merge((channel_y_eq,channel_cr_eq,channel_cb_eq))
    im= cv2.cvtColor(eq_ycrcb, cv2.COLOR_YCrCb2RGB)
    """
    #resize = cv2.resize(rgb, (360, 240))
    #fgmask = fgbg.apply(rgb,rgb,0.01)
    #k= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #opens= cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, k)
    #im = cv2.morphologyEx(opens, cv2.MORPH_CLOSE, k)
    #im = imutils.resize(frame, width = 200 )
    #im= frame
    
    #im= cv2.resize(frame, (360, 240))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
   


    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(frame, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            
            #fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd=hog(im_window, orientations=9, pixels_per_cell=(6,6), cells_per_block=(2, 2),block_norm='L2', visualize=False, transform_sqrt=False, feature_vector=True,multichannel=None)
            
            fd = fd.reshape(1, -1)
            start= time.time()
            pred = clf.predict(fd)
            
            if pred == 1:
                #print(pred)
                if clf.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    #print ("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.01)
    
    
    
    #print ("shape, ", pick.shape)

    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    #cv2.imshow('webcam before nms',im)
    cv2.imshow('webcam after nms',clone)
    end= time.time()
    #print(end-start)
    if cv2.waitKey(10)==27:
        break
    




