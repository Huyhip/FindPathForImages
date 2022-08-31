from math import pow
from math import sqrt
import numpy as np
from matplotlib import cm
import cv2

def pointValue_in(x,y,z,power,xv,yv,zv,values):
    nominator=0
    denominator=0
    for i in range(0,len(values)):
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+(z-zv[i])*(z-zv[i])+5*5)
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def pointValue_out(x,y,z,power,xv,yv,zv,values):
    nominator=0
    denominator=0
    c=1.113195e5
    for i in range(0,len(values)):
        dist = sqrt((c*x-c*xv[i])**2+(c*y-c*yv[i])**2+(z-zv[i])**2)
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def invDist_in(xv,yv,zv,values,zi,xsize=100,ysize=100,power=1):
    xi = np.linspace(0,xsize-1 , xsize)
    yi = np.linspace(0,ysize-1, ysize)
    valuesGrid = np.zeros((ysize,xsize))
    for y in range(0,ysize):
        for x in range(0,xsize):
            valuesGrid[y][x] = pointValue_in(xi[x],yi[y],zi,power,xv,yv,zv,values)
    return valuesGrid


def invDist_out(xv,yv,zv,values,zi,xsize=100,ysize=100,power=1):
    xi = np.linspace(0,xsize-1 , xsize)
    yi = np.linspace(0,ysize-1, ysize)
    valuesGrid = np.zeros((ysize,xsize))
    for y in range(0,ysize):
        for x in range(0,xsize):
            valuesGrid[y][x] = pointValue_out(xi[x],yi[y],zi,power,xv,yv,zv,values)
    return valuesGrid

def NormalizeData(data,max_col):
    return data / max_col

def ConvertRange(old_value,old_min,old_max,new_min,new_max):
    return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

def save(layer,max_col, zi):
    #normalize matrix to (0,1)
    normalized_matrix =  NormalizeData(layer,max_col)
    #print(normalized_matrix)

    #convert from range(0,1) to range (0.5 , 0.9)
    convert_to_range_05_09 = ConvertRange(normalized_matrix,0,1,0.5,0.9)
    #print(convert_to_range_05_09)

    viridis_big = cm.get_cmap('nipy_spectral')
    print("----------")

    #get the value of 'nipy_spectral' color
    matrix_color = viridis_big(convert_to_range_05_09)

    #remove the 4th values which is = 1
    matrix_color2 = matrix_color[:,:,:3]

    #multiple matrix with 255 to get the color RGB value matrix
    matrix_color3 = matrix_color2*255

    #write the matrix to an image
    cv2.imwrite('Images/Img' + str(zi) +'.png', matrix_color3)

    #cv2.imwrite
    #read image
    img = cv2.imread('Images/Img' + str(zi) +'.png')

    #convert image from RGB to BGR
    matrix_color4 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    #save the image
    cv2.imwrite('Images/Img' + str(zi) +'.png', matrix_color4)

def point(img, xsize=512, ysize=256,yr=9.15,xr=4.25):
    x=[]
    y=[]
    for j in range(0,ysize):
        for i in range(0,xsize,10):
            if(img[j,i] == [255,0,0]).all():
                y.append(i)
                x.append(j)
    x = np.array(x)
    y = np.array(y)

    y_in = ConvertRange(NormalizeData(y, xsize), 0, 1, 0, yr)
    x_in = ConvertRange(NormalizeData(x, ysize), 0, 1, 0, xr)

    print(x_in)
    print(y_in)