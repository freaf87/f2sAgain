
#!/usr/bin/python3 -u

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime
import numpy as np  
from f2sCamera import F2SUtils, Camera


def nothing(x):
    pass
    
cv2.namedWindow("Trackbars")
cv2.createTrackbar("min_YCrCb_R", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("min_YCrCb_G", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("min_YCrCb_B", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("max_YCrCb_R", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("max_YCrCb_G", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("max_YCrCb_B", "Trackbars", 255, 255, nothing)
font = cv2.FONT_HERSHEY_COMPLEX

if __name__ == "__main__":
    
    cam = Camera()
    utils = F2SUtils()
    
    while True:
        sourceImage = cam.take_picture()
        mask = utils.maskImage(sourceImage, 0)
        
        # Display the sourceImage & mask image
        cv2.imshow('sourceImage',sourceImage)
        cv2.imshow('mask',mask)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    
