import cv2
import math
import numpy as np   
from PIL import Image

def minimo(r ,g ,b):
    if(r<g and r<b): return r
    if(g<r and g<b): return g
    return b
def maximo(r ,g ,b):
    if(r>g and r>b): return r
    if(g>r and g>b): return g
    return b


def RGB(image_path):
    img = cv2.imread(image_path)
    
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)

    blue = cv2.merge( (b, zeros, zeros) )
    green = cv2.merge( (zeros, g, zeros) ) 
    red =  cv2.merge( (zeros, zeros, r) )
    
    cv2.imshow('Ori', img)
    cv2.imshow('B', blue )
    cv2.imshow('G', green )
    cv2.imshow('R', cv2.merge( (zeros, zeros, r) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return red, green, blue
    

"""def toCMY(imgR,imgG, imgB):    
    imgC = imgR#[[0]*len(img)]*len(img[0])
    imgM = imgG#[[0]*len(img)]*len(img[0])
    imgY = imgB#[[0]*len(img)]*len(img[0])
    #print(imgCMY[0][0])
    for i in range(len(imgR)):
        for j in range(len(imgR[0])):
            imgC[i][j] = (1- (imgR[i][j]/255))
            imgM[i][j] = (1- (imgG[i][j]/255))
            imgY[i][j] = (1- (imgB[i][j]/255))
    #cv2.imshow('cmyk', imgCMY)
    #cv2.waitKey()
    mimi = min(imgC,imgM)
    min_cmy = min(mimi,imgY)
    imgC = (imgC - min_cmy)/(1-min_cmy)
    imgM = (imgM - min_cmy)/(1-min_cmy)
    imgY = (imgY - min_cmy)/(1-min_cmy)
    
    
    cv2.imshow('C', imgC*100)
    cv2.imshow('M', imgM*99.9 )
    cv2.imshow('Y', imgY*99.9 )
    cv2.waitKey()
    
    return imgC, imgM, imgK
"""

    
def CMY(path):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    height,width,ch = img.shape
    img_cmy = np.zeros((height,width,3))

    for i in np.arange(height):
        for j in np.arange(width):
            red = img.item(i,j,0)
            green = img.item(i,j,1)            
            blue = img.item(i,j,2)

            r_ = red/255.
            g_ = green/255.
            b_ = blue/255.

            c = 1 - r_
            m = 1 - g_
            y = 1 - b_

            img_cmy.itemset((i,j,0),int(c*255))
            img_cmy.itemset((i,j,1),int(m*255))
            img_cmy.itemset((i,j,2),int(y*255))

    # mn = min(c, m, y)
    # c = (c-mn) / (1-mn)
    # m = (m-mn) / (1-mn)
    # y = (y-mn) / (1-mn)

    cv2.imwrite('O_cmy.jpg', img_cmy)

    c,m,y = cv2.split(img_cmy)
    zeros = np.zeros_like(c, dtype = float)

    #cv2.imshow('CMY', img_cmy )
    #cv2.imshow('C', cv2.merge( (c, zeros, zeros) ) )
    #cv2.imshow('M', cv2.merge( (zeros, m, zeros) ) )
    cv2.imshow('Y', cv2.merge( (zeros, zeros, y) ) )


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSI(image_path):
    img = cv2.imread(image_path)
    height,width,channel = img.shape
    img_hsi = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Cmin = minimo(r_,g_,b_)
            sum = r_ + g_ + b_

            #theta
            num = (1/2.)*( (r_ - g_) + (r_ - b_) )  # numerador
            dem = math.sqrt( (r_ - g_)**2 + (r_ - b_)*(g_ - b_) )  # denominador
            if(dem!=0): div = num/dem
            else: div = 0
            theta = math.acos(div)
            #Hue
            H = theta
            if(b_ > g_): H = 360-theta 
            #Saturation
            S = 1-(3/sum)*Cmin
            #Intensity
            I = sum/3

            img_hsi.itemset((i,j,0),int(H))
            img_hsi.itemset((i,j,1),int(S))
            img_hsi.itemset((i,j,2),int(I))
    
    cv2.imwrite('O_hsi.jpg', img_hsi)
    h,s,i = cv2.split(img_hsi)
    zeros = np.zeros_like(h, dtype = float)

    cv2.imshow('HSI', img_hsi)
    cv2.imshow('H', h )
    cv2.imshow('S', cv2.merge( (zeros, s, zeros) ) )
    cv2.imshow('I', cv2.merge( (zeros, zeros, i) ) )

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def HSV(image_path):
    img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_hsv = np.zeros((height,width,3))
    

    # CALCULATE
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.
            Cmax = maximo(r_,g_,b_)
            Cmin = minimo(r_,g_,b_)
            delta = Cmax-Cmin

            # Hue Calculation
            if delta == 0:
                H = 0
            elif Cmax == r_ :
                H = 60 * (((g_ - b_)/delta) % 6)
            elif Cmax == g_:
                H = 60 * (((b_ - r_)/delta) + 2)
            elif Cmax == b_:
                H = 60 * (((r_ - g_)/delta) + 4)

            # Saturation Calculation
            if Cmax == 0:
                S = 0
            else :
                S = delta / Cmax
            
            # Value Calculation
            V = Cmax 
            
            # Set H,S,and V to image
            img_hsv.itemset((i,j,0),int(H))
            img_hsv.itemset((i,j,1),int(S))
            img_hsv.itemset((i,j,2),int(V))

    # Write image
    cv2.imwrite('O_hsv.jpg', img_hsv)

    h,s,v = cv2.split(img_hsv)
    zeros = np.zeros_like(h, dtype = float)
    # View image
    cv2.imshow('HSV', img_hsv)
    cv2.imshow('H', h )
    cv2.imshow('S', cv2.merge( (zeros, s, zeros) ) )
    cv2.imshow('V', cv2.merge( (zeros, zeros, v) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def YUV(image_path):
    img = cv2.imread(image_path)
    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #cv2.imshow('jaja', img2)
    
    height,width,channel = img.shape
    print('A')
    img_yuv = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = float(r/255.)
            g_ = float(g/255.)
            b_ = float(b/255.)

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            U = -0.147*r_ -(0.289*g_) + (0.436*b_)
            V = 0.615*r_ - 0.515*g_ - (0.100*b_)
            
           # Y,U,V = 
            """if(Y>=255):
                Y =254
            if(U>=255):
                I = 254
            if(V>=255):
                Q = 254"""

            img_yuv.itemset((i,j,0),int(Y)*255)
            img_yuv.itemset((i,j,1),int(U)*255)
            img_yuv.itemset((i,j,2),int(V)*255)
    
    cv2.imwrite('O_yuv.jpg', img2)

    y,u,v = cv2.split(img_yuv)
    zeros = np.zeros_like(y, dtype = float)

    cv2.imshow('YUV', img_yuv)
    cv2.imshow('Y', y )
    cv2.imshow('U', u )
    cv2.imshow('V', v )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def YIQ(image_path):
    img = cv2.imread(image_path)

    # Get the image's height, width, and channels
    height,width,channel = img.shape

    # Create balnk HSV image
    img_yiq = np.zeros((height,width,3))

    for i in np.arange(height):
        for j in np.arange(width):
            r = img.item(i,j,0)
            g = img.item(i,j,1)
            b = img.item(i,j,2)

            r_ = r/255.
            g_ = g/255.
            b_ = b/255.

            Y = 0.299*r_ + 0.587*g_ + 0.144*b_
            I = 0.596*r_ - 0.275*g_ - 0.321*b_
            Q = 0.212*r_ - 0.523*g_ + 0.311*b_

            img_yiq.itemset((i,j,0),int(Y))
            img_yiq.itemset((i,j,1),int(I))
            img_yiq.itemset((i,j,2),int(Q))
    
    # Write image
    cv2.imwrite('image_yiq.jpg', img_yiq)

    y,i,q = cv2.split(img_yiq)
    zeros = np.zeros_like(y, dtype = float)
    #View image
    cv2.imshow('YIQ', img_yiq)
    cv2.imshow('Y', y )
    cv2.imshow('I', cv2.merge( (zeros, i, zeros) ) )
    cv2.imshow('Q', cv2.merge( (zeros, zeros, q) ) )

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def YCrCb(image_path):
    img = cv2.imread(image_path)
    
    height,width,channel = img.shape
    img_ycrcb = np.zeros((height,width,3))
    for i in np.arange(height):
        for j in np.arange(width):
            R = img.item(i,j,0)
            G = img.item(i,j,1)
            B = img.item(i,j,2)

            Y =   16 +  65.738*R/256. + 129.057*G/256. +  25.064*B/256.
            Cb = 128 -  37.945*R/256. -  74.494*G/256. + 112.439*B/256.
            Cr = 128 + 112.439*R/256. -  94.154*G/256. -  18.285*B/256.

            # print(Y, Cb, Cr)
            
            img_ycrcb.itemset((i,j,0),int(Y))
            img_ycrcb.itemset((i,j,1),int(Cr))
            img_ycrcb.itemset((i,j,2),int(Cb))

    # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 

    y,cr,cb = cv2.split(img_ycrcb)
    zeros = np.zeros_like(y, dtype = float)

    # Write image
    cv2.imwrite('image_ycrcb.jpg', img_ycrcb)

    cv2.imshow('YCrCb', img_ycrcb)
    cv2.imshow('Y', y )
    cv2.imshow('Cr', cr)
    cv2.imshow('Cb', cb )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgname = 'lena.jpg'    
    mR, mG, mB = RGB(imgname)
    #print(len(img),' x ', len(img[0]))
    CMY(imgname)
    HSI(imgname)
    HSV(imgname)
    YUV(imgname)
    YIQ(imgname)
    #YCrCb(imgname)
