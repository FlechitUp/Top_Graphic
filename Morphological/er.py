import cv2
import numpy as np

img = cv2.imread('braim.png',0)
cv2.imshow('Original',img)
kernel = np.ones((5,5),np.uint8)
#Erosion
erosion = cv2.erode(img,kernel,iterations = 1) 
cv2.imshow('er',erosion)
#Closimg
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('er',closing)
#Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('er',opening)

cv2.waitKey()
