
#!/usr/bin/python3 -u

import time
import cv2
import math
import datetime
import numpy as np 
import pytesseract
from picamera.array import PiRGBArray
from picamera import PiCamera 
from sklearn import linear_model
from PIL import Image
    

class Camera():
    def __init__(self):
        self.resolution = (640, 480)
        self.camera = PiCamera(resolution = self.resolution, framerate = 30)
        self.camera.rotation= 180
        self.camera.iso = 100

        # Wait for the automatic gain control to settle
        time.sleep(1)
        
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        g = self.camera.awb_gains
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = g

        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)

    
    def take_picture(self, save=False):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            self.rawCapture.truncate(0)
            if save == True:
                cv2.imwrite("saved/"+str(datetime.datetime.now())+ ".jpg", image)    
            return image
    
class F2SUtils(): 
    def __init__(self):
        self.DEBUG = True
    def maskImage(self,RGBimage, filtertype):
        kernel = np.ones((3, 3), np.uint8)
        if filtertype == 0: # YCrCb Filtering
            min_YCrCb = np.array([0,169,0],np.uint8)
            max_YCrCb = np.array([255, 255, 255],np.uint8)
            imageYCrCb = cv2.cvtColor(RGBimage,cv2.COLOR_BGR2YCR_CB)
            mask = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
            
        else: # HSV Red Filtering
            l_h = 0   
            l_s = 169
            l_v = 123
            u_h = 180 
            u_s = 255 
            u_v = 255
            lower_red = np.array([l_h, l_s, l_v])
            upper_red = np.array([u_h, u_s, u_v])
            hsv = cv2.cvtColor(RGBimage, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_red, upper_red)
            mask = cv2.erode(mask, kernel)  
        
        return mask     
    def findRoadSigns(self, frame):
        
        cropRB   = (17, 85, 603, 169)
        cropStop = (382, 54, 249, 308)

        imCropRoadBlock = frame[int(cropRB[1]):int(cropRB[1]+cropRB[3]), int(cropRB[0]):int(cropRB[0]+cropRB[2])]
        imCropStop = frame[int(cropStop[1]):int(cropStop[1]+cropStop[3]), int(cropStop[0]):int(cropStop[0]+cropStop[2])]
        
        maskRoadblock = self.maskImage(imCropRoadBlock, 0)
        maskStop = self.maskImage(imCropStop, 0)

        if self.DEBUG:
            cv2.imshow("Raw", imCropRoadBlock)
            cv2.imshow("maskRoadblock", maskRoadblock)
            cv2.imshow("maskStop", maskStop)

        isRB, p1, p2 = self.findRoadblock(imCropRoadBlock,maskRoadblock)
        stopSignResult = self.findStopSign(imCropStop, maskStop)
        return isRB, stopSignResult, maskStop
        
    def findRoadblock(self,RGBImage,mask):
        
        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xCoords = []
        yCoords = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                #cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
                M = cv2.moments(contour)
                xCoords.append(int(M['m10']/M['m00']))
                yCoords.append(int(M['m01']/M['m00']))  
            
        # Fit line
        if len(xCoords) > 1:
            ransac = linear_model.RANSACRegressor()
            try:
                ransac.fit(np.asarray(xCoords).reshape(-1, 1),np.asarray(yCoords).reshape(-1, 1))
                inlier_mask = ransac.inlier_mask_
                if (inlier_mask == True).sum() > 3:
                    #remove outliers
                    index = np.where(inlier_mask == False)
                    xCoords = np.delete(xCoords, index)
                    yCoords = np.delete(yCoords, index)
                else:
                    cv2.imwrite("saved/toBehecked_"+str(datetime.datetime.now())+ ".jpg", RGBImage) 
                    return False, ((),),((),) #if (inlier_mask == True).sum() > 3:
                    
            except ValueError:
                pass
                #print("have a closer look. Ransac skipped. (Maybe perfect match) ")
                
            # sort x coordinates
            sortedIndex = sorted(range(len(xCoords)), key=lambda k: xCoords[k])
            xCoords = [xCoords[i] for i in sortedIndex]
            yCoords = [yCoords[i] for i in sortedIndex]

            diff = 0
            # Check if distance of center are equidistant
            try:
                count = 0
                oldDist = 0
                for i in range(len(xCoords)):
                    dist = math.sqrt((xCoords[i+1] - xCoords[i])*(xCoords[i+1] - xCoords[i]) + (yCoords[i+1] - yCoords[i])*(yCoords[i+1] - yCoords[i]))
                    compDist = abs((dist - oldDist)/dist)

                    if i > 0:
                        if compDist < 0.3:
                            count = count+1
                        else:
                            count = count -1
                            if count < 0:
                                count = 0
                    oldDist = dist 
                    #print("DEBUG: ", count)
            except IndexError:
                pass
            return count >=1, (xCoords[0],yCoords[0]), ((xCoords[-1],yCoords[-1]))
        else:
            cv2.imwrite("saved/toBehecked_"+str(datetime.datetime.now())+ ".jpg", RGBImage) 
            return False,((),),((),)
            
    def findStopSign(self,RGBImage,mask):
        #result = pytesseract.image_to_string(Image.fromarray(mask))
        return "" #result
        """
        edges = cv2.Canny(mask, 100, 200)
        template_paths = {'stop': './edges_stop.png'}
        edges_templates = {}
        for name, template_path in template_paths.items():
            template = cv2.imread(template_path, 0)
            edges_templates[name] = template 
        return self.template_matching(edges_input, edges_templates)
        """
    def templateMatching(self,input_image, templates):
        
        result_templates = {}

        for name, template in templates.items():
            tH = template.shape[0]
            tW = template.shape[1]

            gray = input_image.copy()
            found = None

            gH = gray.shape[0]
            gW = gray.shape[1]

            iteration = 5

            if name == 'stop':
                start_w = int((0.25 * gW))
                gray = gray[:,start_w:  ]
                iteration = 10
                #cv2.imwrite("saved/"+str(datetime.datetime.now())+ "_stop.jpg", gray)
            elif name == 'barrier':
                start_h = int((0.35 * gH))
                #end_h = int((0.75 * tW))
                gray = gray[start_h:,:]
                iteration = 5
                #cv2.imwrite("saved/"+str(datetime.datetime.now())+ "_barrier.jpg", gray)
                

            # loop over the scales of the image
            for scale in np.linspace(0.2, 1.0, iteration)[::-1]:
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing
                resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(edged, template, cv2.TM_CCORR)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

            
            (maxVal, _, _) = found
            
            result_templates[name] = maxVal
        
        print("Stop: " + str(result_templates['stop']))
        print("Barrier: " + str(result_templates['barrier']))
        if result_templates['stop'] > 9000000 and result_templates['barrier'] > 7000000:
            return 'both'
        elif result_templates['barrier'] > 7000000:
            return 'barrier'
        elif result_templates['stop'] > 9000000:
            return 'stop'

        return None
