import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv

def remodificarValue(value):
    #print("ok" ,value)
    if (value < 0):     
        return int(0)
    elif(value > 254): 
        value = 254 
        return int(254)

def laplace(img):    
    G = [0,1,0,1,-4,1,0,1,0]
    
    tam=3
    rows = np.size(img, 0)
    cols = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = int(img[i][j+1]) + int(img[i][j-1]) + int(img[i+1][j]) + int(img[i-1][j]) - 4*int(img[i][j])
            if(val <0): val = 0
            elif (val>255): val = 255
            M[i][j] = val
    
    cv.imshow('Laplace', M)
    cv.waitKey()
    return M
#def a(img):


def roberts(img):
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = -1*int(img[i-1][j-1]) + int(img[i][j])
            if(val < 0): val = 0
            elif(val>254): val = 254
            Gx[i][j] = val
            val2 = -1*int(img[i-1][j]) + int(img[i][j-1])
            if(val2 < 0): val2 = 0
            elif(val2>254): val2 = 254
            Gy[i][j] = val2
        
    new_image = Gx + Gy
    cv.imshow('Roberts', new_image)
    cv.waitKey()     


def sobel(img):
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = int(img[i-1][j-1]) - int(img[i-1][j+1]) + 2*int(img[i][j-1]) - 2*int(img[i][j+1]) + int(img[i+1][j-1]) - int(img[i+1][j+1])           
            if(val < 0): val = 0
            elif( val >254 ): val = 254            
            Gx[i][j] = val
            val2 = int(img[i-1][j-1]) - int(img[i+1][j-1]) + 2*int(img[i-1][j]) - 2*int(img[i+1][j]) + int(img[i-1][j+1]) - int(img[i+1][j+1])
            if(val2 < 0): val2 = 0
            elif( val2 >254 ): val2 = 254   
            Gy[i][j] = val2
    
    new_image = Gx + Gy 
    cv.imshow('Sobel', new_image)
    cv.waitKey() 

def prewitt(img):
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = -1*int(img[i-1][j-1]) + int(img[i-1][j+1]) - int(img[i][j-1]) + int(img[i][j+1]) - int(img[i+1][j-1]) + int(img[i+1][j+1])
            if(val < 0): val = 0
            elif(val>255): val = 255
            Gx[i][j] = val
            val2 = -1*int(img[i-1][j-1]) + int(img[i+1][j-1]) - int(img[i-1][j]) + int(img[i+1][j]) - int(img[i-1][j+1]) + int(img[i+1][j+1])
            if (val2<0): val2 =0
            elif(val2>254): val2 = 254
            Gy[i][j] = val2
        
    new_image = Gx + Gy   
    cv.imshow('Prewitt', new_image)
    cv.waitKey() 
 
if __name__ == '__main__':
    img = cv.imread('ximeequ.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('Origi', img)
    cv.waitKey()
    
    #laplace(img)    
    #roberts(img)
    sobel(img)
    #prewitt(img)
    
