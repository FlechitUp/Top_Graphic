"""import numpy as np
import cv2 as cv
#from PIL import Image


def gcd(m, n):
    if n == 0:
        return m
    if m > n:
        return gcd(n, m - n)
    else:
        return gcd(m, n-m)

def multByEulerian(i,j):
    suma = i +j 
    if suma >
 tff(nameIMG):
    j = -1
    img = cv.imread(nameIMG, cv.IMREAD_GRAYSCALE)

    rows, cols = img.shape[:2]
    imgDuplic = np.zeros((rows * 2, cols * 2), dtype=np.uint8)
    
    #cv.imshow('Equalized ', img)
    #cv.waitKey()

    #print('Type ', type(img[0][0]), ' , ', type(imgDuplic[0][0]))
    #img.resize(rows, cols * 2)
    arrZer = [0] * (cols * 2)

    for i in range(0,rows):
        for j in range(0, cols):
            #print(img[i][j])
            imgDuplic[i][j] =  img[i][j]
    
    cv.imshow('TF ', imgDuplic)
    cv.waitKey()


    for i in range(0,rows*2):
        for j in range(0, cols*2):            
            imgDuplic[i][j] = (imgDuplic[i][j] * pow(j, i + j)) #% 4294967295
            #print(imgDuplic[i][j])
   


    

if __name__ == '__main__':
    nameImg = 'coins.png'
    tff(nameImg)
"""
#convolucion
#transformada
#inversa
import math
import cmath

#escala de grises
import numpy as np
import time
from PIL import Image
import copy


"""-
 -   -  -
-      -
"""


def fourier(n):
    F = 1 / n
    y = [0, 0, 0, 1, 1, 1]  # 0,0.7,1,0.7,0,-0.7,-1,-0.7]#regresa
    yN = []
    sumT = ""
    sumR = 0
    sumI = 0
    j = -1
    for u in range(0, n):
        #print("u: "),
        #print(u)
        sumR = 0
        sumI = 0
        for i in range(0, n):
            #print("i: "),
            #print(i)
            #print (y[i])
            sumR += y[i] * math.cos((2 * math.pi * u * i) / n)
            sumI += y[i] * math.sin((2 * math.pi * u * i) / n)

        sumT = str(sumR) + " + " + str(sumI) + "j"
        yN.append(math.sqrt(sumR * sumR + sumI * sumI))
        print(sumT)
    print(yN)


def fourier2D(n, m):
    T = Image.open('bw.jpg').convert('L')

    n = np.size(T, 0)
    m = np.size(T, 1)

    T = np.array(T)

    #yN= np.zeros((n, m ), dtype=np.uint8)
    yN = np.zeros_like(T, np.uint8)

    sumR = 0
    sumI = 0
    j = -1

    for u in range(0, n):
        for v in range(0, m):
            sumT = 0
            for x in range(0, n):
                sumR = 0
                sumI = 0
                for y in range(0, m):
                    sumR += int(T[x][y]) * math.cos(2 *
                                                    math.pi * ((u * x) / n + (v * y) / m))
                    sumI += int(T[x][y]) * math.sin(2 *
                                                    math.pi * ((u * x) / n + (v * y) / m))

                #sumT = str(sumR) + " + "+str(sumI)+"j"
                sumT += math.sqrt(sumR * sumR + sumI * sumI)

            yN[u][v] = (sumT)

    print(yN)

    img = Image.fromarray(yN)
    img.show()


T = Image.open('coins.png').convert('L')
T.show()
n = np.size(T, 0)
print('n', n)
m = np.size(T, 1)
yNT = [[complex(0) for x in range(n)] for y in range(m)]
mNT = [[complex(0) for x in range(n)] for y in range(m)]

#yNT= np.zeros_like(T, np.uint32)


G = [1, 4, 7, 4, 1,
     4, 16, 26, 16, 4,
     7, 26, 41, 26, 7,
     4, 16, 26, 16, 4,
     1, 4, 7, 4, 1]


def Fourier(T):
    T = np.array(T)
    mN = np.zeros_like(T, np.uint32)
    #agregar mascara a la imagen
    for z in range(0, 5):
        for y in range(0, 5):
            mN[z][y] = G[z * 5 + y]
    print (mN)
    yN = np.zeros_like(T, np.uint32)
    yNT[0][245]=80
    print(len(yNT),' , ',len(yNT[0]))

    for u in range(0, n-1):  # N
        for v in range(0, m-1):  # M
            sumT = complex(0)
            sumM = complex(0)
            for x in range(0, n):  # N
                for y in range(0, m):  # M
                    e = cmath.exp(-1j * math.pi * 2.0 *
                                  ((float(u * x) / n) + (float(v * y) / m)))
                    sumT += e * complex(T[x][y])
                    sumM += e * complex(mN[x][y])
            try:
                 yNT[u][v] = (sumT / (n * m))
            except IndexError:                    
                print('u ',u, '; v', v)
            yN[u][v] = (sumT.real / (n * m))
            mNT[u][v] = (sumM / (n * m))
            mN[u][v] = (sumM.real / (n * m))

    print('ok')
    img = Image.fromarray(yN)
    img.show()
    #img.save('bwF.png')


def convolucion():
    #multiplicacion punto a punto
    for z in range(0, n):
        for y in range(0, m):
            yNT[z][y] = yNT[z][y] * mNT[z][y]


def InvFourier(T):
    print('inversa')
    T = np.array(T)
    yN = np.zeros_like(T, np.uint32)

    for u in range(0, n):  # N
        for v in range(0, m):  # M
            sumT = complex(0)
            for x in range(0, n):  # N
                for y in range(0, m):  # M
                    e = cmath.exp(1j * math.pi * 2.0 *
                                  ((float(u * x) / n) + (float(v * y) / m)))
                    sumT += e * complex(yNT[x][y])
            yN[u][v] = (sumT.real)
    print(yN)
    img = Image.fromarray(yN)
    img.show()


#fourier2D(6,6)#,img)
Fourier(T)
convolucion()
InvFourier(T)
