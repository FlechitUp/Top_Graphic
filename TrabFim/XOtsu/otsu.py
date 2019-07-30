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


def globalOtsu(img, inicio = 1, fin = 255):
    #cv2.imshow('GLOOOOO', img)
    #cv2.waitKey()
    height = np.size(img, 0)
    width = np.size(img, 1)

    TotalPixels = len(img.ravel())  # Or height * width
    print('TotalPixels ',TotalPixels)
    #print('Imicio ',inicio, ' fim ', fin)

    maxVal = max(img.ravel())
    Gcolors = [0] * 256;
    for i in range(0, len(img.ravel())):
        Gcolors[int(img.ravel()[i])] += 1

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
    for i in range(0, height):
    	for j in range(0, width):
        	if img[i][j] < prct:        	
        		img[i][j] = 0.0
        	else:
        		img[i][j] = 0.99  #Blamco
    cv2.imshow('GLOOOOO', img)
    cv2.waitKey()
    return img

    #BIMARIZATIOM

    """
    pathImg2 = 'subimg.jpg'
    cv2.imwrite(pathImg2, img)
    img2 = Image.open(pathImg2).convert('L')

    Ah = np.size(img2, 0)
    Aw = np.size(img2, 1)
    NI = np.zeros_like(img2, np.uint8)
    A = np.asarray( img2 )
    #print(A)
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]>bestThreshold:	
                NI[i][j]=0
            else:
                NI[i][j]=255

    img1 = Image.fromarray(NI)
    img1.show()
    return img1"""

def mediana(img,mask):
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

def localOtsu(imgpath):
    #img = data.imread(imgpath, as_grey=True)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE) 
    cv2.imshow('Mediana', img)
    cv2.waitKey()

    radius = 3
    selem = disk(radius)
    #local_otsu = rank.otsu(img, selem, out=None)
    local_otsu = threshold_local(img,15,'median') #threshold_mean
    binary_local = img > local_otsu
    print("Local Otsu ",len(local_otsu))
    binary_local = Image.fromarray(local_otsu)
    binary_local.show()
    return binary_local

def sobel(img):
    rows, cols = img.shape[:2]

    Gx = img.copy()
    Gy = img.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = int(img[i-1][j-1]) - int(img[i-1][j+1]) + 2*int(img[i][j-1]) - 2*int(img[i][j+1]) + int(img[i+1][j-1]) - int(img[i+1][j+1])           
            
            if(val < 0): val = 0
            elif( val >255 ): val = 255            
            Gx[i][j] = val

            val2 = int(img[i-1][j-1]) - int(img[i+1][j-1]) + 2*int(img[i-1][j]) - 2*int(img[i+1][j]) + int(img[i-1][j+1]) - int(img[i+1][j+1])
            if(val2 < 0): val2 = 0
            elif( val2 >255 ): val2 = 255   
            Gy[i][j] = val2
    
    new_image = Gx + Gy 
    #cv2.imshow('Sobel', new_image)
    #cv2.waitKey()
    return new_image

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
    
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img



def localStd(img):
    #img = cv2.imread(imgPath, True)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img)
    img = img / 255.0

    # c = imfilter(I,h,'symmetric');
    h = np.ones((5,5))
    n = h.sum()
    n1 = n - 1
    c1 = cv2.filter2D(img**2, -1, h/n1, borderType=cv2.BORDER_REFLECT)
    c2 = cv2.filter2D(img, -1, h, borderType=cv2.BORDER_REFLECT)**2 / (n*n1)
    J = np.sqrt( np.maximum(c1-c2,0) )

    cv2.imshow('stdfilt', J)
    cv2.waitKey(0)
    cv2.destroyWindow('stdfilt') 
    #cv2.imwrite('stda.jpg', J)
    return J

def interseccion(img,img2):
    img3 = img.copy()
    print(img[0][0])
    print('------------')
    print(img2[0][0])
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img2, np.uint8)
    A = np.asarray( img )
    B = np.asarray( img2 )
    print(A)
    print(B)
    for i in range(0,Ah):
        for j in range(0,Aw):
            img3[i][j] = 255#0
            if img[i][j]==img2[i][j]:
                img3[i][j] = img[i][j] #255
            """if A[i][j]==B[i][j]:
                NI[i][j]=A[i][j]
                print('iiiii ', A[i][j])"""
    cv2.imshow('AND', img3)
    cv2.waitKey(0)
    #print (NI)
    #img1 = Image.fromarray(NI)
    #img1.save('my.jpg')
    #img1.show()
    return img3

def erosion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])

    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                contA=0
                contM=0
                for k in range(0,Mh):
                    for l in range(0,Mw):
                        if M[k][l]==255:
                            contM+=1
                            if i+k<Ah and j+l<Aw:
                                if A[i+k][j+l]==255:
                                    contA+=1
                if contA==contM:
                    #if NI[i+k][j+l]!=255:
                    #if NI[i+x][j+y]!=255:
                    NI[i+x][j+y]=255

    img1 = Image.fromarray(NI)
    #img1.save('my.png')
    img1.show()   
    return img1

def cruzMasks(n):
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,n):
        l=[]
        q=int(n/2)
        if i == int(n/2) :
            for i in range (0,n):
                l.append(255)
        else:
            for j in range(0,q):
                l.append(0)
            l.append(255)
            for k in range(0,q):
                l.append(0)
        cuadrado.append(l)

    #print (cuadrado)
    return cuadrado

if __name__ == '__main__':
    pathImg = 'H07.png'#'H02.png'#'2im.jpg'#3.bmp

    img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)   
    plt.hist(img.ravel(),256,[0,256], density=True )
    
    height, width = img.shape
### ------------------------------------------ Part 1 ----------------------------------------- """
##########  Local Otsu ###################
    #imgLO = mediana(img,9)
    #pathImg2 = 'lotsu.jpg'
    #cv2.imwrite(pathImg2, imgLO)
    #lcimg=localOtsu(pathImg)

### ------------------------------------------ Part 2 ----------------------------------------- """
    ###### Sobel #########
    img = sobel(img)
    cv2.imshow('Sobel', img)
    
    ###### Local Standard Deviation #####
    img = localStd(img)

    ##### Global Otsu #####
    imgLeft = globalOtsu(img,66,120)    
			
### ----------------------------------------- Part 3 --------------------------------------------
    local = cv2.imread('LocalOtsu.jpg', cv2.IMREAD_GRAYSCALE)   
    imt = interseccion(local, imgLeft)
    M = cruzMasks(3)
    erosion(imt,M,1,1)