import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv

def horizontal(img):    
    G = [-1,-1,-1,2,2,2,-1,-1,-1]
  
    tam =3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]           
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
    
    #cv.imshow('horizontal', M)
    #cv.waitKey()
    return M
#def a(img): 


def vertical(img):    
    G = [-1,2,-1,-1,2,-1,-1,2,-1]
    tam =3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]           
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1

    
    #cv.imshow('verti', M)
    #cv.waitKey()
    return M

def line45Plus(img):    
    G = [-1,-1,2,-1,2,-1,2,-1,-1]
    
    tam =3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]           
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
    
    #cv.imshow('+45', M)
    #cv.waitKey()
    return M

def line45Minus(img):    
    G = [2,-1,-1,-1,2,-1,-1,-1,2]

    tam =3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]           
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
    
    #cv.imshow('-45', M)
    #cv.waitKey()
    return M

def poimts(img):    
    G = [-1,-1,-1,-1,8,-1,-1,-1,-1]

    tam =3
    height = np.size(img, 0)
    width = np.size(img, 1)
    var=(tam*tam)/2
    var=var/tam
    vary=var%tam
    M = img.copy()
    img = np.asarray( img)
    M = np.asarray( M)
    M = np.zeros_like(img, np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            sum1=0
            for z in range(0,tam):
                for y in range(0,tam):
                    if(i+z<height and j+y<width):
                        sum1+=img[i+z][j+y]*G[z*3+y]           
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
    
    #cv.imshow('poimts', M)
    #cv.waitKey()
    return M

if __name__ == '__main__':
    img = cv.imread('church1.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('Origi', img)
    cv.waitKey()
    
    
    M = vertical(img)
    H = horizontal(img)        
    l45 = line45Minus(img)
    l45m = line45Plus(img)
    P=poimts(img)
    cv.imshow('All', M+H+l45+l45m+P)
    cv.waitKey()
