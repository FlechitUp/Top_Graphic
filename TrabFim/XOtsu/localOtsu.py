import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import cv2
import math
import numpy as np   
from PIL import Image

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.filters import threshold_local, threshold_mean
#from skimage.util import img_as_ubyte
from skimage import data, io

def medianaA(img,mask):
    cv2.imshow('Original', img)
    rows,cols=img.shape
    new_image = img.copy()
    dif=int(mask/2)
    for i in range(dif, rows-dif):
        for j in range(dif, cols-dif):
            lista=[]
            for x in range(i-dif,i+dif+1):
                for y in range(j-dif,j+dif+1):
                    lista.append(int(img[x][y]))
            lista.sort()
            new_image[i][j]=lista[int((mask*mask)/2)]
    cv2.imshow('Mediana', new_image)
    cv2.waitKey()
    return new_image

def getWeights(threshold, gcolorsVc, TotalPxls):  #TotalPxls -> # of pixels of Image
	global totalB
	global totalF
	totalB = 0
	totalF = 0
	weightB = 0
	weightF = 0

	for i in range(0, threshold):
		totalB += gcolorsVc[i]
	weightB = totalB/TotalPxls

	for i in range(threshold, len(gcolorsVc)):
		totalF += gcolorsVc[i]

	weightF = totalF/TotalPxls

	return (weightB, weightF)

def getMeans(threshold,gcolorsVc):
	sumB = 0
	sumF = 0
	#print('totals ', totalB, totalF)

	for i in range(0, threshold):
		sumB += (i*gcolorsVc[i])
	if totalB == 0 :
		sumB = 0	
	else:
		sumB /=totalB

	for i in range(threshold, len(gcolorsVc)):	
		sumF += (i*gcolorsVc[i])
	
	if totalF == 0:
		sumF = 0
	else:
		sumF /= totalF

	return (sumB, sumF)

def betweenClassVariance(Wb, Ub, Wf, Uf):
	#print( 'Wb',Wb,' ', 'uB', Ub)
	u = (Wb*Ub)+(Wf*Uf)	

	return (Wb*Wf)*pow(Ub-u,2)


def globalOtsu(img, posIh, posIw, height, width, inicio = 1, fin = 255):    
    #height, width = img.shape

    TotalPixels = (height - posIh)*(width - posIw ) #len(img.ravel())  # Or height * width
    print('TotalPixels ',TotalPixels)
    #print('Imicio ',inicio, ' fim ', fin)
    
    #maxVal = max(img.ravel())
    Gcolors = [0] * 256;
    #for i in range(0, len(img.ravel())):
    #    Gcolors[int(img.ravel()[i])] += 1
    for i in range(posIh, height):
    	for j in range(posIw, width):
    		Gcolors[int(img[i][j])] +=1

    plt.plot(Gcolors , color='blue' )  #Histogram
    plt.xlabel('Total Pixels '+str(TotalPixels) )

    MaxVariance = 0
    bestThreshold = inicio
    for i in range(inicio, fin):
        Wb, Wf = getWeights(i, Gcolors, TotalPixels)  # Wb + Wf must  be = 1.0 
        #print(Wb + Wf)
        Ub, Uf = getMeans(i,Gcolors)
        variance = betweenClassVariance(Wb,Ub, Wf, Uf)
        #print (i, ' -> ', bc)
        #print('vari ',variance)
        if variance > MaxVariance:
            bestThreshold = i
            MaxVariance = variance
    
    print('Best threshold = ',bestThreshold,' ',MaxVariance)
    plt.axvline(x=bestThreshold,color='red') #Adding a vertical line for threshold
    plt.title('Best threshold = ' + str(bestThreshold))
    plt.show()
    percemtageThreshold = (bestThreshold/255)
    prct =np.float64(percemtageThreshold)
    print('Percemtage threshold = ',prct)

    ##############################################################################
    print(type(img[0][0]))
    print(type(percemtageThreshold))
    for i in range(posIh, height):
    	for j in range(posIw, width):
        	"""if img[i][j] < prct:        	
        		img[i][j] = 0.0
        	else:
        		img[i][j] = 0.99  #Blamco"""
        	if img[i][j] <bestThreshold:
        		img[i][j] = 255
        	else:
        		img[i][j] = 0

    #cv2.imwrite('LocalOtsu.jpg', img)
    cv2.imshow('Local img', img)
    cv2.waitKey()    
    return img


if __name__ == '__main__':
    pathImg = 'H07.png'#'H02.png'#'2im.jpg'

    img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)   
    #plt.hist(img.ravel(),256,[0,256], density=True )
    
    height, width = img.shape

    img2 = img.copy()

    imgLO = medianaA(img2,3)
    imgLeft = globalOtsu(imgLO,0,0,int(height/2),int(width/2))  #S1
    imgLeft = globalOtsu(imgLeft,0,int(width/2), int(height/2),int(width))	#S2
    imgLeft = globalOtsu(imgLeft,int(height/2),0, height,int(width/2))	#S3
    imgLeft = globalOtsu(imgLeft,int(height/2),int(width/2), height,width)  #S4
    #imgLeft = imgLeft / 255.0
    cv2.imwrite('LocalOtsu.jpg', imgLeft)
    print(imgLeft)
		
    #imgLeft = globalOtsu(img,0,0, height,width)  #S4

