import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

img = cv2.imread('2imGRAY.jpg', cv2.IMREAD_GRAYSCALE)
height, width = img.shape

print("******** Info **********")
print("Size:")
print("Height = ", height)
print("wWdth = ", width)
print("************************")

x, bins, p = plt.hist(img.ravel(),256,[0,256], density=True)
#for item in p:
#    item.set_height(item.get_height()/sum(x))
print(x[120])
plt.show()

#Calcular probabilidades 
prob_k = []
for i in x:
	prob_k.append(i/(height*width))

#print(prob_k)

#Calcular sumas acumulativas
p1_k = []
for i in range(0,len(x)):
	val = 0 
	for j in range(0,i):
		val +=prob_k[j]
	p1_k.append(val)

#print(p1_k)

#Umbral T = 90

#Calcular medias acumulativas
m_k = []
for i in range(0,len(x)):
	val = 0 
	for j in range(0,i):
		val += j*prob_k[j]
	m_k.append(val)

#print(m_k)


#Calcular media global
M_G = 0
for i in range(0,len(x)):
	M_G += i*prob_k[i]

#print(M_G)

#Calcular varianza entre clases
o_b2_k = []

