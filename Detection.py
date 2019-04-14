#!/usr/bin/python3 -u

import time
import cv2
from f2sCamera import F2SUtils, Camera

if __name__ == "__main__":
    
    cam = Camera()
    F2sUtils = F2SUtils()
    while True:
        sourceImage = cam.take_picture()
        isRB, Stop , mask = F2sUtils.findRoadSigns(sourceImage)
        print("RB/StopSign found ? ", isRB, Stop)
        
        key = cv2.waitKey(1)
        if key == 27:
            cv2.imwrite("mask.png", mask)
            break
        
    
