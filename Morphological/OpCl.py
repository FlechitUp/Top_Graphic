import numpy as np
import time
from PIL import Image
import math

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
    #img.save('my.png')
    img1.show()
    #print (NI)
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
    img1.show()
    #print (NI)
    return img1

if __name__ == '__main__':
    img = Image.open('j.png').convert('L')
    #Mask
    M=cuadradoMask(3)
   
    #print(M)
    
    #dilatacion + 
    #img2=dilatacion(img,M,1,1)
    #erosion -
    
    #img2=dilatacion(img,M,1,1)
    img2=erosion(img,M,1,1)





