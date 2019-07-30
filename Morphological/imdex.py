import numpy as np
import time
from PIL import Image
import math

#operaciones morfologicas

#masks:
def cuadradoMask(n):#numero impar
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,n):
        l=[]
        for j in range(0,n):
            l.append(255)
        cuadrado.append(l)
    print (cuadrado)
    return cuadrado

def diamanteMask(n):#numero impar
    origen=n/2 # (2,2)
    cuadrado = []
    for i in range(0,int(n/2)+1):
        l=[]
        q=int(n/2-i)
        for j in range(0,q):
            print (j)
            l.append(0)
        for o in range(q,n-q):
            l.append(255)
        for k in range(0,q):
            l.append(0)
        cuadrado.append(l)
    for i in range (len(cuadrado)-1,-1,-1):
        cuadrado.append(cuadrado[i])

    print (cuadrado)
    return cuadrado

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
####################
def binarizar(img):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    print(A)
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]>200:#255/2:
                NI[i][j]=255
            else:
                NI[i][j]=0
    print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()
    return img1

def diferencia(img,img2):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    B = np.asarray( img2 )
    print(A)
    print(B)
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]!=B[i][j]:
                NI[i][j]=A[i][j]
    #print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()
    return img1

#M2 mas grande
def diferenciaMask(M,M2):
    NI = M2
    for i in range(0,len(M2)):
        for j in range(0,len(M2)):
            if i<len(M) and j <len(M[0]) and  M[i][j]!=M2[i][j]:
                NI[i][j]=M2[i][j]
    #print (NI)
    return NI

def interseccion(img,img2):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    B = np.asarray( img2 )
    print(A)
    print(B)
    for i in range(0,Ah):
        for j in range(0,Aw):
            if A[i][j]==B[i][j]:
                NI[i][j]=A[i][j]
    print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()
    return img1

def complemento(img):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    print(A)
    for i in range(0,Ah):
        print(A[i])

        for j in range(0,Aw):
            if A[i][j]==255:
                NI[i][j]=0
            else:
                NI[i][j]=255

    #print (NI)
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()
    return img1

#x,y origen
def dilatacion(img,M,x,y):
    Ah = np.size(img, 0)
    Aw = np.size(img, 1)
    NI = np.zeros_like(img, np.uint8)
    A = np.asarray( img )
    Mh=len(M)
    Mw=len(M[0])

    """Ah=5
    Aw=5
    Mh=3
    Mw=3
    NI=[]
    for i in range (0,Ah):
        q=[]
        for j in range(0,Aw):
            q.append(0)
        NI.append(q)
    """
    for i in range (0,Ah):
        for j in range(0,Aw):
            #if i + Mh-1<Ah and j+Mw-1<Aw:
            if i+x<Ah and j+y<Aw:
                if A[i+x][j+y]==255:
                    for k in range(0,Mh):
                        for l in range(0,Mw):
                            if M[k][l]==255:
                                if i+k<Ah and j+l<Aw:
                                    if NI[i+k][j+l]!=255:
                                        NI[i+k][j+l]=255
    img1 = Image.fromarray(NI)
    #img.save('my.png')
    #img1.show()    
    return img1

#x,y origen
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
    #img1.show()   
    return img1

# (a erosiion b) dilatacion b
def opening(img, MCru):
	img2=erosion(img,MCru,1,1)
	img3=dilatacion(img2,MCru,1,1)
	img3.show()

# (a dilatacion b) erosion b
def closing(img):
	img2=dilatacion(img,M,2,2)
	img3=erosion(img2,M,2,2)
	print("//////")
	print(M)
	#img3.show()
	return img3

#(a erosion b) interseccion (a complemento erosion B)
def hit_miss(img):
	img2=erosion(img,M,1,1)
	img3=complemento(img)
	M3=diferenciaMask(M,M2)
	img4=erosion(img3,M3,2,2)
	img = diferencia(img2,img4)
	img.show()
	return img


if __name__ == '__main__':
    #img = Image.open('cuadrado.jpg').convert('L')
    img = Image.open('j.png').convert('L')
    #img = Image.open('piso.jpg').convert('L')


    #M=cruzMasks(5)
    M = cuadradoMask(3)
    M2 = diamanteMask(5)
    MCru = cruzMasks(15)
    #M=diamanteMask(3)
    
    #print(M)
    
    """
    img=[[0,0,0,0,0],
         [0,255,255,0,0],
         [255,255,255,0,255],
         [0,255,255,0,0],
         [0,0,0,0,0]
    ]"""
    #img.show()
    img=binarizar(img)
    

    erosiom =erosion(img,MCru,1,1)
    
    erosiom.show()
    #dil = dilatacion(img,M,2,2)
    #dil.show()
    
    #opening(img,MCru)
    #clo = closing(img)	
    #clo.show()
    #hit_miss(img)
  