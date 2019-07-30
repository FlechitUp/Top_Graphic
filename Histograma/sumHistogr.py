# python maim.py  n
# Bar Chart
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv



def histogram(imgA, imgB):
    rows, cols = imgA.shape[:2]   
    tonos = 256
    histogram = [0] * tonos
    s_k = [0] * tonos

    for i in range(rows):
        for j in range(cols):
            histogram[img[i][j]] += 1
            histogram[imgB[i][j]] += 1

#    for i in histogram:
        

    """plt.plot(histogram, color='blue')
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()"""
    hist = cv.calcHist([imgA], [0], None, [256], [0, 256])
    #print("histograma", hist)
    plt.plot(hist, color='gray')
    #plt.show()
    
    hist2 = cv.calcHist([imgB], [0], None, [256], [0, 256])
    #print("histograma", hist)
    plt.plot(hist2, color='red')
    #plt.show()

    plt.plot(histogram, color='blue')   # es la suma de los dos histogramas
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()


def increase_brightness(img, value=30):
    img = cv.UMat(cv.imread(img, cv.IMREAD_COLOR))
    imgUMat = cv.UMat(img)
    hsv = cv.cvtColor(imgUMat, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    imgUMat = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

    cv.imshow("edges", imgUMat)
    cv.waitKey()

    return imgUMat


if __name__ == '__main__':
    img = cv.imread('cameraman.jpg', cv.IMREAD_GRAYSCALE)
    imgb = cv.imread('woman.jpg', cv.IMREAD_GRAYSCALE)
    
    #histogram(img, imgb)
    frame = increase_brightness('cameraman.jpg', value=180)
    plt.close('all')
