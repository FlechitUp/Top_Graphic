import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import cv2
import math
import numpy as np   
from PIL import Image
  

def getGreen(image_path):
    img = cv2.imread(image_path)
    
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)

    blue = cv2.merge( (b, zeros, zeros) )
    green = cv2.merge( (zeros, g, zeros) ) 
    red =  cv2.merge( (zeros, zeros, r) )
        
    #cv2.imshow('B', blue )
    cv2.imshow('Green', green )
    #cv2.imshow('R', cv2.merge( (zeros, zeros, r) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return green

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
    cv2.imshow('Sobel', new_image)
    cv2.waitKey()




if __name__ == '__main__':
    img = cv2.imread('2im.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Origi', img)
    cv2.imwrite('2imGRAY.jpg', img)
    cv2.waitKey()
    #sobel(img)
    greeImg = getGreen('2imGRAY.jpg')
    plt.hist(img.ravel(),256,[0,256], density=True )
    plt.show()
    
    img = cv2.medianBlur(img,5)
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    plt.subplot(2,2,2),plt.imshow(th1,'gray')
    plt.title('Adaptive Mean Thresholding')
    plt.xticks([]),plt.yticks([])
    plt.show()
