# python maim.py  n
# Bar Chart
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def histogram(img):
    rows, cols = img.shape[:2]
    tonos = 256
    histogram = [0] * tonos
    s_k = [0] * tonos
    maxim = 0
    prevMax = 0
    pos = 0
    
    
    
    for i in range (rows):
        for j in range (cols):
            histogram[img[i][j]]+=1

    
    plt.plot(histogram , color='blue' )
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    

    for i in range(0,len(histogram)):        
        maxim = max(maxim, histogram[i])
        if (maxim == histogram[i]):
            pos = i
        
    print('Pico max', pos)
    maxim = 0
    histogram[pos] = 0 
    pos2 = 0
    for i in range(0,len(histogram)):        
        if maxim < histogram[i]:
            pos2 = i
            maxim = histogram[i]
        #maxim = max(maxim, histogram[i])
        #if (maxim == histogram[i] and pos != pos2):
            #pos2 = i    
    print('Pico max-1', pos2)
    

    cv.imshow('Original', img)
    cv.waitKey()

    pr_k = histogram
    for i in range( len(pr_k)):
        pr_k[i] = float(pr_k[i]/float(rows * cols))   #average          
        for j in reversed(range(i+1)):        
            s_k[i] +=pr_k[j]
        s_k[i] = int(np.round(s_k[i] * 255,0))
            
    plt.plot(s_k , color='red' )
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()
    
    for i in range(len(s_k)):
        print(i , " | ", s_k[i])

    
    image2 = img.copy()
    for i in range (rows):
        for j in range (cols):
            image2[i][j] = s_k[img[i][j]]
        
    cv.imwrite('ximeequ.jpg',image2)
    cv.imshow('Equalized ', image2)
    cv.waitKey()
    

if __name__ == '__main__':
    img = cv.imread('image.jpg', cv.IMREAD_GRAYSCALE)
    img2 = np.asarray(img)    
    
    histogram(img)
    plt.close('all')

