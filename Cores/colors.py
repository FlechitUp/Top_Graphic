def RGB(image_path):
    img = cv2.imread(image_path)
    # rows, cols = img.shape[:2]
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)

    # blue = img[]
    
    blue = cv2.merge( (b, zeros, zeros) )
    green = cv2.merge( (zeros, g, zeros) ) 
    red =  cv2.merge( (zeros, zeros, r) )
    
    cv2.imshow('Original', img)
    cv2.imshow('blue', blue )
    cv2.imshow('green', green )
    cv2.imshow('red', red )

    # for i in range(1, rows-1):
    #     for j in range(1, cols-1):
    cv2.waitKey()
    return red, green, blue

cmyk_scale = 100

def rgb_to_cmyk(r,g,b):
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale


"""def BIGCMYK(mR, mG, mB):
    
    for i in range(len(mR)):
        for j in range(len(mR[0])):
            mR[i][j] =
            mG[i][j] =
            mB[i][j] =
   """         
        
#img = cv2.imread('lena.jpg')
