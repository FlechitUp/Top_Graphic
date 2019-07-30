# python maim.py  n
# Bar Chart
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv


def suavizado(img, valMask):
    rows, cols = img.shape[:2]
    tonos = 256
    histogram = [0] * tonos
    s_k = [0] * tonos
    
    cemtro = math.floor(valMask/2)
    sum1 = 0

    for i in range (0,rows-valMask):
        for j in range (0,cols-valMask):
            sum1 = 0
            for p in range(valMask):
                for q in range(valMask):
                    sum1 +=int(img[i+p][j+q])                
            img[cemtro+i][cemtro+j] = (sum1/(pow(valMask,2)))
    
    cv.imshow('Suavizado',img)
    cv.waitKey()
    cv.destroyAllWindows()
    

def media(img, valMask): # media ponderada
    rows, cols = img.shape[:2]
    tonos = 256
    histogram = [0] * tonos
    s_k = [0] * tonos
    cemtro = math.floor(valMask/2)
    sum1 = 0

    for i in range (1,rows-valMask):
        for j in range (1,cols-valMask):
            sum1 = 0 
            for p in range(valMask):
                for q in range(valMask):
                    if(cemtro+i == i+p) and(cemtro+j == j+q): sum1 += int(2*img[i+p][j+q]) 
                    else:sum1 += int(img[i+p][j+q])                  
            img[cemtro+i][cemtro+j] =(sum1/((valMask*valMask)+1)) 
    
    cv.imshow('image',img)
    cv.waitKey()
    cv.destroyAllWindows()


def mediana2(img, valMask):
    new_image = img.copy()
    rows, cols = img.shape[:2]
    rows = rows - valMask +1
    cols = cols - valMask + 1
    for i in range(0,rows):
        for j in range(0,cols): 
            sum = 0
            tmp = []
            for k in range(0, valMask):
                for l in range(0,valMask):
                    row = i+k 
                    col = j+l
                    tmp.append(img[col][row])

            tmp = sorted(tmp)
            
    
def mediana(img, valMask):
    rows, cols = img.shape[:2]
    new_image = img.copy()
    cemtro = math.floor(valMask/2)
    sum1 = 0

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            sum1 = 0 
            for p in range(valMask):
                for q in range(valMask):
                    mask = []
                    mask.append(img[i-1][j-1])
                    mask.append(img[i][j-1])
                    mask.append(img[i+1][j-1])
                    mask.append(img[i-1][j])
                    mask.append(img[i][j])
                    mask.append(img[i+1][j])
                    mask.append(img[i-1][j+1])
                    mask.append(img[i][j+1])
                    mask.append(img[i+1][j+1])

                    mask = sorted(mask)
                    new_image[i][j] = mask[5]
    
    cv.imshow('Mediana', new_image)
    cv.waitKey()
    
def max_min(img,m, valMask):  # m = 0 ->max ,m=1 -> min
    rows, cols = img.shape[:2]
    img2 = img.copy()
        
    if m== 0:
        window_name= "MAX"
        pos= 8
    else:    
        window_name = "Min"
        pos= 0
        

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            mask = []
            mask.append(img[i-1][j-1])
            mask.append(img[i][j-1])
            mask.append(img[i+1][j-1])
            mask.append(img[i-1][j])
            mask.append(img[i][j])
            mask.append(img[i+1][j])
            mask.append(img[i-1][j+1])
            mask.append(img[i][j+1])
            mask.append(img[i+1][j+1])

            mask= sorted(mask)
            img2[i][j] = mask[pos]
    
    cv.imshow(window_name, img2)
    cv.waitKey()
    
"""def roberts(img):
    rows, cols = img.shape[:2]
    img2 = img.copy()
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
   """
def gauss(img):
    G = [1,2,1,2,4,2,1,2,1]
    
    tam=3
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
            sum1=sum1/(16)
            if(i+tam<height and j+tam<width):
                if sum1<0:
                    M[int(i+var)][int(j+vary)]=0
                else: 
                    M[int(i+var)][int(j+vary)]=sum1
   
    cv.imshow('Gauss', M)
    cv.waitKey()
    return M

if __name__ == '__main__':
    img = cv.imread('image.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('Origi', img)
    cv.waitKey()
    img2 = np.asarray(img)
    suavizado(img,11)
    #media(img,11)
    #mediana(img,5)
    #max_min(img,0,5)
    #gauss(img)
