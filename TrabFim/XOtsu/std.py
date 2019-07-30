import cv2
import numpy as np

imgPath = '2im.jpg'

def localStd(imgPath):
    img = cv2.imread(imgPath, True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0

    # c = imfilter(I,h,'symmetric');
    h = np.ones((3,3))
    n = h.sum()
    n1 = n - 1
    c1 = cv2.filter2D(img**2, -1, h/n1, borderType=cv2.BORDER_REFLECT)
    c2 = cv2.filter2D(img, -1, h, borderType=cv2.BORDER_REFLECT)**2 / (n*n1)
    J = np.sqrt( np.maximum(c1-c2,0) )

    cv2.imshow('stdfilt', J)
    cv2.waitKey(0)
    cv2.destroyWindow('stdfilt') 
    return J

localStd(imgPath)